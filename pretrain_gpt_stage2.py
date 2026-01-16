# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
import copy
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
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from mindspeed_llm.training.training import pretrain
from megatron.core.transformer.spec_utils import import_module
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

    # Stage-2 student model only uses transformer layers (no embedding/output head)
    # Teacher is built lazily after initialization in forward_step_stage2.
    # Keep globals for teacher and a buffer if needed.
    global global_teacher
    # Do not instantiate teacher here; distributed is ready after initialize_megatron.
    # We'll build teacher on first forward.
    if 'global_teacher' not in globals():
        global_teacher = None

    if not args.use_legacy_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        # Student: disable output head; feed decoder_input directly and train to predict next hidden state.
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=False,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=None,
        )
        # Avoid unused-parameter buckets when PP=1: embeddings aren't used in stage2 (we pass decoder_input),
        # so drop their grads to prevent DDP grad-reduce assertions.
        if pre_process:
            for p in model.embedding.parameters():
                p.requires_grad_(False)
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        # Legacy path: student transformer-only
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=False
        )
        if pre_process:
            for p in model.embedding.word_embeddings.parameters():
                p.requires_grad_(False)

    # Optional projection head for InfoNCE on hidden states; trains jointly with the student.
    if getattr(args, 'hidden_loss_type', 'l2') == 'info_nce':
        # proj_dim = getattr(args, 'hidden_proj_dim', config.hidden_size)
        # proj_layers = max(1, getattr(args, 'hidden_proj_layers', 1))
        # in_dim = config.hidden_size
        # layers = []
        # for _ in range(proj_layers - 1):
        #     layers.append(torch.nn.Linear(in_dim, proj_dim))
        #     layers.append(torch.nn.GELU())
        #     in_dim = proj_dim
        # layers.append(torch.nn.Linear(in_dim, proj_dim))
        # model.hidden_projector = torch.nn.Sequential(*layers)
        # debug: 设置其为恒等映射
        model.hidden_projector = torch.nn.Identity()

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


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

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
    try:
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not args.enable_elastic_training or not elastic_training_common.zit_scale_in_running_state():
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    except ImportError:
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
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

    if args.use_legacy_models:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, loss_mask=loss_mask)

    return output_tensor, partial(loss_func, loss_mask)


def _ensure_teacher_initialized():
    """Build and load the frozen teacher GPTModel that outputs hidden states (no LM head)."""
    global global_teacher
    if global_teacher is not None:
        return
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    # Teacher config mirrors student config, but with embedding enabled and output head disabled
    if args.yaml_cfg is not None:
        teacher_config = core_transformer_config_from_yaml(args, "language_model")
    else:
        teacher_config = core_transformer_config_from_args(args)

    if args.spec is not None:
        teacher_layer_spec = import_module(args.spec)
    else:
        if use_te:
            teacher_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
        else:
            teacher_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

    teacher_config = copy.deepcopy(teacher_config)
    if hasattr(args, 'teacher_num_layers') and args.teacher_num_layers is not None:
        setattr(teacher_config, 'num_layers', args.teacher_num_layers)

    teacher = GPTModel(
        config=teacher_config,
        transformer_layer_spec=teacher_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=True,
        post_process=False,  # return decoder hidden states
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
    )
    teacher.cuda(torch.cuda.current_device())  # GPU allocation

    # Load teacher checkpoint from args.teacher_load using the same helper as pretrain_gpt,
    # but avoid mutating global args/rerun state by snapshotting and restoring.
    if not hasattr(args, 'teacher_load') or args.teacher_load is None:
        raise ValueError("--teacher-load must be specified for stage2 training")

    args_snapshot = copy.deepcopy(args.__dict__)
    rerun_snapshot = None
    try:
        rerun_snapshot = get_rerun_state_machine().state_dict(data_iterator=None, ckpt_format=args.ckpt_format)
    except Exception:
        rerun_snapshot = None

    # Temporarily disable loading of rng/optim and checkpoint-arg overrides
    orig_flags = (
        getattr(args, 'no_load_rng', False),
        getattr(args, 'no_load_optim', False),
        getattr(args, 'finetune', False),
    )
    args.no_load_rng = True
    args.no_load_optim = True
    args.finetune = True  # prevents checkpoint args from overwriting current args

    # Wrap in list to match checkpoint loader expectations; unwrap after load.
    ddp_like_teacher = [teacher]
    assert os.path.exists(getattr(args, 'teacher_load')), f"teacher checkpoint path {getattr(args, 'teacher_load')} does not exist!"
    _ = load_checkpoint(ddp_like_teacher, None, None, load_arg='teacher_load', strict=False, allow_ckpt=True)
    teacher = unwrap_model(ddp_like_teacher)[0]

    # Restore args to snapshot
    for k, v in args_snapshot.items():
        setattr(args, k, v)
    # Restore rerun state if it was modified
    if rerun_snapshot is not None:
        try:
            get_rerun_state_machine().load_state_dict(rerun_snapshot)
        except Exception:
            pass
    # Restore flags
    args.no_load_rng, args.no_load_optim, args.finetune = orig_flags

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    global_teacher = teacher


def _hidden_loss_func(per_token_loss: torch.Tensor, loss_mask: torch.Tensor):
    """Aggregate per-token hidden-state prediction loss with mask and reduce across DP."""
    args = get_args()
    # Flatten per-token loss and apply mask
    loss_mask = loss_mask.view(-1).float()
    losses = per_token_loss.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses * loss_mask).view(1), total_tokens.view(1)])

    # Reduce for logging across data-parallel group
    reporting_loss = loss.clone().detach()
    try:
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not args.enable_elastic_training or not elastic_training_common.zit_scale_in_running_state():
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    except ImportError:
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {'hidden loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step_stage2(data_iterator, model: GPTModel):

    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    _ensure_teacher_initialized()
    teacher = global_teacher

    with torch.no_grad():
        # Teacher returns decoder hidden states since post_process=False
        h = teacher(tokens, position_ids, attention_mask)
        h = h.detach()  # keep [t, b, h]

    # 构造学生输入与标签
    student_inp = h[:-1]            # h[t-1]
    target = h[1:]                  # h[t]
    # Student: feed hidden states as decoder_input; position_ids used for RoPE
    pred_h = model(
        decoder_input=student_inp,
        input_ids=None,
        position_ids=position_ids[:, :-1],
        attention_mask=attention_mask[:, :-1, :-1] if attention_mask else None,
    )
    # Per-token loss over valid positions at t>=1
    masked_loss_mask = loss_mask[:, 1:]
    if args.hidden_loss_type == 'info_nce':
        if not hasattr(unwrap_model(model), 'hidden_projector'):
            raise RuntimeError("hidden_projector missing; enable --hidden-loss-type info_nce only when projector is built")

        # Shared projector to reduce dimensionality before computing InfoNCE
        hidden_projector = unwrap_model(model).hidden_projector
        pred_z = F.normalize(hidden_projector(pred_h), dim=-1)   # [T-1, B, D]
        tgt_z = F.normalize(hidden_projector(target), dim=-1)    # [T-1, B, D]

        # Time-step grouped negatives: at each time t, only tokens across batch at that step participate.
        # masked_loss_mask is [B, T-1]; convert to time-major for alignment with pred_z/tgt_z
        time_major_mask = masked_loss_mask.transpose(0, 1).bool()      # [T-1, B]
        per_token_loss = torch.zeros_like(time_major_mask, dtype=pred_z.dtype)  # [T-1, B]

        temperature = args.hidden_proj_temperature
        for t in range(pred_z.size(0)):
            valid = time_major_mask[t]  # [B]
            if valid.sum() < 1:
                continue
            pred_t = pred_z[t][valid]   # [N, D]
            tgt_t = tgt_z[t][valid]     # [N, D]

            logits = torch.matmul(pred_t, tgt_t.transpose(0, 1)) / temperature  # [N, N]
            log_probs = F.log_softmax(logits, dim=-1)
            diag_losses = -log_probs.diag()  # [N]

            per_token_loss[t][valid] = diag_losses

        # Convert back to batch-major to match downstream masking
        per_token_loss = per_token_loss.transpose(0, 1)  # [B, T-1]
    else:
        per_token_loss = (pred_h - target).pow(2).mean(dim=-1)  # [T-1, B]
        per_token_loss = per_token_loss.transpose(0, 1)         # [B, T-1]
    return per_token_loss, partial(_hidden_loss_func, loss_mask=loss_mask[:, 1:])


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
    """Add stage2-specific arguments for teacher loading and loss selection."""
    parser.add_argument('--teacher-load', type=str, default=None,
                        help='Checkpoint path for the frozen teacher GPTModel used to emit hidden states.')
    parser.add_argument('--teacher-num-layers', type=int, default=None,
                        help='Number of transformer layers in stage1 teacher; overrides config for teacher.')
    parser.add_argument('--hidden-loss-type', type=str, default='l2', choices=['l2', 'info_nce'],
                        help='Loss type for hidden-state prediction.')
    parser.add_argument('--hidden-proj-dim', type=int, default=256,
                        help='Projection dimension used before InfoNCE loss.')
    parser.add_argument('--hidden-proj-layers', type=int, default=2,
                        help='Number of linear layers in the InfoNCE projector (>=1).')
    parser.add_argument('--hidden-proj-temperature', type=float, default=0.07,
                        help='Softmax temperature for InfoNCE logits.')
    return parser


def main():
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step_stage2,
             extra_args_provider=_extra_args_provider)


if __name__ == "__main__":
    main()
