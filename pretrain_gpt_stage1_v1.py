# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
from functools import partial
from typing import Union

import torch.nn.functional as F

import torch
from mindspeed_llm import megatron_adaptor
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from mindspeed_llm.training.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import unwrap_model
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from mindspeed_llm.training.utils import  set_mtp_batch_list, get_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank


class VQCodebook(torch.nn.Module):
    """Minimal VQ-VAE codebook for hidden discretization.

    Supports two update modes for the codebook:
    - 'grad': standard gradient-based codebook update using codebook loss (sg[z] - e)^2
    - 'ema': VQ-VAE(EMA) with exponential moving average updates; only commitment loss contributes to gradients
    """

    def __init__(self, num_codes: int, code_dim: int, beta: float,
                 update_method: str = 'ema', ema_decay: float = 0.99, ema_eps: float = 1e-5):
        super().__init__()
        self.codebook = torch.nn.Embedding(num_codes, code_dim)
        self.beta = beta
        self.update_method = update_method
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        # Track per-code assignment counts across steps
        self.register_buffer('code_usage_counts', torch.zeros(num_codes, dtype=torch.float64))
        self.register_buffer('current_usage_counts', torch.zeros(num_codes, dtype=torch.float64))

        # EMA statistics (same dtype/device as embedding by default)
        init_value = 1e-4 
        self.register_buffer('ema_cluster_size', torch.full((num_codes,), init_value))
        self.register_buffer('ema_embed_avg', self.codebook.weight.data.clone() * init_value)

        # Disable gradients to codebook in EMA mode
        self.codebook.weight.requires_grad = (self.update_method == 'grad')

    def forward(self, z: torch.Tensor):
        # z: [S, B, H]; flatten to [N, H]
        flat = z.reshape(-1, z.size(-1))
        # Compute nearest code via L2 distance
        # dist = ||z||^2 + ||e||^2 - 2 z e^T
        z_sq = torch.sum(flat * flat, dim=1, keepdim=True)          # [N, 1]
        e = self.codebook.weight                                    # [K, H]
        e_sq = torch.sum(e * e, dim=1)                              # [K]
        logits = - (z_sq - 2 * torch.matmul(flat, e.t()) + e_sq)    # negative dist
        idx = torch.argmax(logits, dim=1)                           # [N]
        z_q = F.embedding(idx, e).view_as(z)                        # [S, B, H]

        # Track usage counts (no grad)
        with torch.no_grad():
            hist = torch.bincount(idx, minlength=self.codebook.num_embeddings).float()
            self.code_usage_counts += hist
            self.current_usage_counts = hist

        # Loss terms
        # commitment: beta * ||z - sg[e]||^2 (always applied)
        loss_commit = self.beta * F.mse_loss(z, z_q.detach(), reduction='sum') / z.shape[-1]

        if self.update_method == 'grad':
            # codebook update via gradients: ||sg[z] - e||^2
            loss_cb = F.mse_loss(z.detach(), z_q, reduction='sum') / z.shape[-1]
            vq_loss = loss_cb + loss_commit
        else:
            # EMA update: update codebook weights without gradients
            with torch.no_grad():
                K = self.codebook.num_embeddings
                # one-hot assignments [N, K]
                onehot = F.one_hot(idx, num_classes=K).to(flat.dtype)
                # cluster sizes [K]
                cluster_size = onehot.sum(dim=0)
                # sum of embeddings per code [K, H]
                embed_sum = torch.matmul(onehot.t(), flat)
                # update exponential moving averages
                self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size * (1 - self.ema_decay))
                self.ema_embed_avg.mul_(self.ema_decay).add_(embed_sum * (1 - self.ema_decay))
                # normalize and update codebook weights
                denom = self.ema_cluster_size.unsqueeze(1) + self.ema_eps
                updated = self.ema_embed_avg / denom
                
                # only update codes used in the current batch
                active_mask = (cluster_size > 0)
                if torch.any(active_mask):
                    self.codebook.weight.data[active_mask] = updated.to(self.codebook.weight.dtype)[active_mask]

                # reseed codes that have never been used across all steps
                never_used_mask = (self.code_usage_counts == 0)
                inactive_idx = torch.nonzero(never_used_mask, as_tuple=False).view(-1)
                if inactive_idx.numel() > 0 and flat.numel() > 0:
                    num_reseed = min(inactive_idx.numel(), flat.size(0))
                    # sample without replacement from current batch projections
                    rand_perm = torch.randperm(flat.size(0), device=flat.device)[:num_reseed]
                    reseed_vectors = flat[rand_perm].to(self.codebook.weight.dtype)
                    self.codebook.weight.data[inactive_idx[:num_reseed]] = reseed_vectors
            vq_loss = loss_commit

        # Straight-through estimator
        z_out = z + (z_q - z).detach()
        return z_out, vq_loss


class VQHookContext:
    """Holds hook state per rank/stage."""

    def __init__(self, codebook: VQCodebook, target_layer_number: int):
        self.codebook = codebook
        self.target_layer_number = target_layer_number
        self.handle = None
        self.vq_loss = None
        self.code_usage_counts = None
        self.current_usage_counts = None
        self.vq_grad_norm = None

    def reset(self):
        self.vq_loss = None
        self.code_usage_counts = None
        self.current_usage_counts = None
        self.vq_grad_norm = None

    def attach(self, model: GPTModel):
        # Only attach when the target layer lives on this pipeline stage.
        if not hasattr(model, 'decoder'):
            return False
        for layer in getattr(model.decoder, 'layers', []):
            if getattr(layer, 'layer_number', None) == self.target_layer_number:
                def _hook(_, inputs, outputs):
                    # outputs: (hidden, context)
                    hidden = outputs[0]
                    quantized, vq_loss = self.codebook(hidden)
                    self.vq_loss = vq_loss
                    self.code_usage_counts = getattr(self.codebook, 'code_usage_counts', None)
                    self.current_usage_counts = getattr(self.codebook, 'current_usage_counts', None)
                    return (quantized, outputs[1])

                self.handle = layer.register_forward_hook(_hook)
                return True
        return False

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    # VQ settings (stage1) from args/env; keep defaults no-op when disabled
    vq_enable = getattr(args, 'vq_enable', False)
    vq_layer_idx = getattr(args, 'vq_layer_idx', None)
    vq_codebook_size = getattr(args, 'vq_codebook_size', None)
    vq_beta = getattr(args, 'vq_beta', 0.25)
    vq_update = getattr(args, 'vq_update', 'ema')
    vq_ema_decay = getattr(args, 'vq_ema_decay', 0.99)
    vq_ema_eps = getattr(args, 'vq_ema_eps', 1e-5)
    vq_code_dim = getattr(args, 'vq_code_dim', None)

    if not args.use_legacy_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )

    # Attach VQ hook only when enabled and all params are set
    if vq_enable:
        # Resolve defaults
        if vq_layer_idx is None:
            vq_layer_idx = config.num_layers  # by default last layer of the block
        if vq_codebook_size is None:
            vq_codebook_size = getattr(args, 'padded_vocab_size', 32000)
        if vq_code_dim is None:
            vq_code_dim = config.hidden_size

        # Build hook context and register on matching layer_number
        codebook = VQCodebook(
            vq_codebook_size, vq_code_dim, vq_beta,
            update_method=vq_update,
            ema_decay=vq_ema_decay,
            ema_eps=vq_ema_eps,
        ).to(torch.cuda.current_device())
        codebook = codebook.to(config.params_dtype)
        ctx = VQHookContext(codebook, vq_layer_idx)
        attached = ctx.attach(model)
        if not attached:
            print_rank_0(f"[VQ] target layer {vq_layer_idx} not on this stage; VQ disabled on this rank")
            ctx = None
    else:
        ctx = None

    # Stash context on model for forward_step to access
    model._vq_ctx = ctx
    model._vq_enable = vq_enable

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()

    is_middle_stage = not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage())
    pretrain_not_tnd_flags = not args.is_instruction_dataset and not args.reset_attention_mask
    if pretrain_not_tnd_flags and is_middle_stage:
        return (None,) * 5

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    if args.return_document_ids and mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0:
        print("current idx: {}, current rank: {}, data_parallel_rank: {}, document_ids: {}".format(batch['idx'], torch.distributed.get_rank(), mpu.get_data_parallel_rank(), batch['document_ids']))
        batch.pop('document_ids', None)
        batch.pop('idx', None)

    # get batch_list for mtp_block
    if args.mtp_num_layers:
        mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
        set_mtp_batch_list(mtp_batch_list)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tuple):
    """Loss function with optional VQ regularizer and code usage metrics."""

    args = get_args()

    output_tensor, vq_loss, vq_hist, vq_current_hist = output_tuple

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    lm_loss_val = torch.sum(losses.view(-1) * loss_mask)

    # Combine with VQ loss (already scalar)
    combined = lm_loss_val + vq_loss
    loss = torch.cat([combined.view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    lm_reporting = torch.tensor([lm_loss_val.detach()], device=loss.device)
    vq_reporting = torch.tensor([vq_loss.detach()], device=loss.device)

    # Reduce usage histogram across DP to report utilization; only if provided
    vq_hist_reporting = None
    if vq_hist is not None:
        vq_hist_reporting = vq_hist.detach()
        vq_current_hist_reporting = vq_current_hist.detach()
        vq_hist_reporting = vq_hist_reporting.to(loss.device)
        vq_current_hist_reporting = vq_current_hist_reporting.to(loss.device)
    try:
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not args.enable_elastic_training or not elastic_training_common.zit_scale_in_running_state():
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(lm_reporting, group=mpu.get_data_parallel_group())
            torch.distributed.all_reduce(vq_reporting, group=mpu.get_data_parallel_group())
            if vq_hist_reporting is not None:
                torch.distributed.all_reduce(vq_hist_reporting, group=mpu.get_data_parallel_group())
            if vq_current_hist_reporting is not None:
                torch.distributed.all_reduce(vq_current_hist_reporting, group=mpu.get_data_parallel_group())
    except ImportError:
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(lm_reporting, group=mpu.get_data_parallel_group())
        torch.distributed.all_reduce(vq_reporting, group=mpu.get_data_parallel_group())
        if vq_hist_reporting is not None:
            torch.distributed.all_reduce(vq_hist_reporting, group=mpu.get_data_parallel_group())
        if vq_current_hist_reporting is not None:
            torch.distributed.all_reduce(vq_current_hist_reporting, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    metrics = {
        'lm loss': (lm_reporting[0], reporting_loss[1]),
        'vq loss': (vq_reporting[0], reporting_loss[1]),
    }

    # Add utilization stats if histogram available
    if vq_hist_reporting is not None:
        used_codes = (vq_hist_reporting > 0).float().sum()
        total_codes = torch.tensor(vq_hist_reporting.numel(), device=vq_hist_reporting.device, dtype=torch.float64)
        metrics['vq total util'] = (used_codes, total_codes)
    if vq_current_hist_reporting is not None:
        used_codes_current = (vq_current_hist_reporting > 0).float().sum()
        total_codes_current = torch.tensor(vq_current_hist_reporting.numel(), device=vq_current_hist_reporting.device, dtype=torch.float64)
        metrics['vq current util'] = (used_codes_current, total_codes_current)
    return (
        loss[0].clone(),
        local_num_tokens,
        metrics,
    )


def forward_step_stage1_v1(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    # reset VQ state
    vq_ctx = getattr(unwrap_model(model), '_vq_ctx', None)
    if vq_ctx is not None:
        vq_ctx.reset()

    if args.use_legacy_models:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, loss_mask=loss_mask)

    # collect vq loss and histogram if present on this rank
    if vq_ctx is not None and vq_ctx.vq_loss is not None:
        vq_loss = vq_ctx.vq_loss
        vq_hist = vq_ctx.code_usage_counts
        vq_current_hist = vq_ctx.current_usage_counts
    else:
        device = output_tensor.device if torch.is_tensor(output_tensor) else torch.cuda.current_device()
        vq_loss = torch.zeros(1, device=device)
        vq_hist = None
        vq_current_hist = None

    return (output_tensor, vq_loss, vq_hist, vq_current_hist), partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def _extra_args_provider(parser):
    """Add stage1 VQ-related args (kept minimal to avoid core changes)."""
    parser.add_argument('--vq-enable', action='store_true', help='Enable VQ codebook on a transformer layer')
    parser.add_argument('--vq-layer-idx', type=int, default=None, help='Layer number (1-based) to attach VQ codebook; defaults to last layer')
    parser.add_argument('--vq-codebook-size', type=int, default=None, help='Number of codes in the VQ codebook; defaults to padded_vocab_size')
    parser.add_argument('--vq-code-dim', type=int, default=None, help='Dimensionality of codes; defaults to hidden_size')
    parser.add_argument('--vq-beta', type=float, default=0.25, help='Commitment loss weight beta')
    parser.add_argument('--vq-update', type=str, default='ema', choices=['ema', 'grad'],
                        help='Codebook update method: EMA (recommended) or gradient-based')
    parser.add_argument('--vq-ema-decay', type=float, default=0.99, help='EMA decay for codebook stats')
    parser.add_argument('--vq-ema-eps', type=float, default=1e-5, help='EMA epsilon for normalization')
    return parser


def main():
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step_stage1_v1,
             extra_args_provider=_extra_args_provider)


if __name__ == "__main__":
    main()
