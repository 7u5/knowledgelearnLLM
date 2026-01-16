# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import math
from functools import partial
from typing import Any, Optional, Union, Tuple
from torch import Tensor

import torch
import torch.nn.functional as F
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
from megatron.core.utils import make_viewless_tensor
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params
from megatron.core.models.gpt import GPTModel
from mindspeed_llm.training.training import pretrain
from megatron.core.transformer.spec_utils import import_module, ModuleSpec
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
from mindspeed_llm.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.jit import jit_fuser


def sinkhorn_knopp_batched(A, it=1000, eps=1e-8):
    """Batched Sinkhorn-Knopp to project onto doubly-stochastic matrices."""
    batch_size, n, _ = A.shape
    device = A.device
    u = torch.ones(batch_size, n, device=device)
    v = torch.ones(batch_size, n, device=device)

    for _ in range(it):
        v_temp = v.unsqueeze(2)  # (B, n, 1)
        Av = torch.bmm(A, v_temp).squeeze(2)  # (B, n)
        u = 1.0 / (Av + eps)

        u_temp = u.unsqueeze(2)  # (B, n, 1)
        At_u = torch.bmm(A.transpose(1, 2), u_temp).squeeze(2)  # (B, n)
        v = 1.0 / (At_u + eps)

    U = torch.diag_embed(u)  # (B, n, n)
    V = torch.diag_embed(v)  # (B, n, n)
    P = torch.bmm(torch.bmm(U, A), V)
    return P, U, V


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        mean = (x ** 2).mean(-1, keepdim=True)
        out_mean = x / torch.sqrt(mean + self.eps)
        return self.gamma * out_mean


class ManifoldHyperConnectionFuse(torch.nn.Module):
    """Minimal mHC from notebook adapted for [B, L, N, D] tensors."""

    def __init__(self, dim, rate, layer_id, max_sk_it):
        super().__init__()
        self.n = rate
        self.dim = dim
        self.nc = self.n * self.dim
        self.n2 = self.n * self.n
        self.norm = RMSNorm(dim * rate)  # norm flatten
        
        # parameters
        self.w = torch.nn.Parameter(torch.zeros(self.nc, self.n2 + 2 * self.n))
        self.alpha = torch.nn.Parameter(torch.ones(3) * 0.01)
        self.beta = torch.nn.Parameter(torch.zeros(self.n2 + 2 * self.n) * 0.01)
        
        # max sinkhorn knopp iterations
        self.max_sk_it = max_sk_it

    def mapping(self, h, res_norm):
        B, L, N, D = h.shape

        # 1.vectorize
        h_vec_flat = h.reshape(B, L, N * D)

        # RMSNorm Fused Trick: gamma-scaling
        h_vec = self.norm.gamma * h_vec_flat

        # 2.projection
        H = h_vec @ self.w

        # RMSNorm Fused: compute r
        r = h_vec_flat.norm(dim=-1, keepdim=True) / math.sqrt(self.nc)
        r_ = 1.0 / r

        # 4. mapping
        n = N
        H_pre = r_ * H[:, :, :n] * self.alpha[0] + self.beta[:n]
        H_post = r_ * H[:, :, n:2 * n] * self.alpha[1] + self.beta[n:2 * n]
        H_res = r_ * H[:, :, 2 * n:] * self.alpha[2] + self.beta[2 * n:]

        # 5. final constrained mapping 
        H_pre = torch.sigmoid(H_pre)
        H_post = 2 * torch.sigmoid(H_post)

         # 6. sinkhorn_knopp iteration
        H_res = H_res.reshape(B, L, N, N)
        H_res_exp = H_res.exp()
        with torch.no_grad():
            _, U, V = res_norm(H_res_exp.reshape(B * L, N, N), self.max_sk_it)
        P = torch.bmm(torch.bmm(U.detach(), H_res_exp.reshape(B * L, N, N)), V.detach())
        H_res = P.reshape(B, L, N, N)

        return H_pre, H_post, H_res

    def process(self, h: torch.Tensor, H_pre: torch.Tensor, H_res: torch.Tensor):
        h_pre = H_pre.unsqueeze(dim=2) @ h
        h_pre.squeeze_(dim=2)
        h_res = H_res @ h
        return h_pre, h_res

    def depth_connection(self, h_res, h_out, beta):
        post_mapping = beta.unsqueeze(dim=-1) @ h_out
        return post_mapping + h_res


def _bias_dropout_add_func_mhc(x_with_bias, residual, prob, training, mHC_Hpost):
    # type: (Tuple[Tensor, Optional[Tensor]], Tensor, float, bool, Tensor) -> Tensor
    # NOTE: Previously, the argument `bias` used to be passed as
    # `bias.expand_as(residual)` when the `bias_dropout_func` is called from the
    # transformer layer but broadcasting should automatically take care of that.
    # Also, looking at broadcasting semantics, `expand_as` and broadcasting
    # seem to be identical performance-wise (both just change the view).

    x, bias = x_with_bias  # unpack

    # If we want to train mixed precision, then the output of this function
    # should be half precision. However, in AMP O1, the input (residual) is
    # in fp32, and it will up-cast the result to fp32, causing pipeline parallel
    # GPU communication to hang. Therefore, we need to cast residual to the same
    # dtype as x.
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)

    # The Dropout operation, Residual Addition and the tensor returning can be
    # done generically outside the if statement, but that stops fusing of Bias
    # Addition-Dropout-Residual Addition operation. So doing it together inside
    # the conditional branch to improve performance
    if bias is not None:
        x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = mHC_Hpost.unsqueeze(dim=-1) @ out.unsqueeze(dim=-2)
        out = residual + out
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = mHC_Hpost.unsqueeze(dim=-1) @ out.unsqueeze(dim=-2)
        out = residual + out
        return out


@jit_fuser
def bias_dropout_add_fused_train_mHC(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float, mHC_Hpost: torch.Tensor
) -> torch.Tensor:
    return _bias_dropout_add_func_mhc(x_with_bias, residual, prob, True, mHC_Hpost)


@jit_fuser
def bias_dropout_add_fused_inference_mHC(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]], residual: torch.Tensor, prob: float, mHC_Hpost: torch.Tensor
) -> torch.Tensor:
    return _bias_dropout_add_func_mhc(x_with_bias, residual, prob, False, mHC_Hpost)


def bias_dropout_add_unfused_mHC(training):
    def _bias_dropout_add(x_with_bias, residual, prob, mHC_Hpost):
        return _bias_dropout_add_func_mhc(x_with_bias, residual, prob, training, mHC_Hpost)

    return _bias_dropout_add


def get_bias_dropout_add_mHC(training, fused):
    if fused:
        # jit scripting for a nn.module (with dropout) is not
        # triggering the fusion kernel. For now, we use two
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if training:
            return bias_dropout_add_fused_train_mHC
        else:
            return bias_dropout_add_fused_inference_mHC
    else:
        return bias_dropout_add_unfused_mHC(training)


class mHCTransformerLayer(TransformerLayer):
    """Drop-in replacement for TransformerLayer with mHC in MLP path."""

    def __init__(
        self, 
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config,
                         submodules=submodules,
                         layer_number=layer_number,
                         hidden_dropout=hidden_dropout)
        
        args = get_args()

        self.attn_mhc = ManifoldHyperConnectionFuse(
            dim=config.hidden_size,
            rate=args.mhc_rate,
            layer_id=layer_number,
            max_sk_it=args.mhc_max_sk_it
        )
        self.mlp_mhc = ManifoldHyperConnectionFuse(
            dim=config.hidden_size,
            rate=args.mhc_rate,
            layer_id=layer_number,
            max_sk_it=args.mhc_max_sk_it
        )

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        pre_mlp_layernorm_output, residual, context, mHC_H_post_mlp = self._forward_attention(*args, **kwargs)
        output = self._forward_mlp(pre_mlp_layernorm_output, residual, mHC_H_post_mlp)
        if self.layer_number == get_args().num_layers:
            output = output.sum(dim=2)  # over mHC dimension
        return output, context

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        args = get_args()
        inference_context = deprecate_inference_params(inference_context, inference_params)

        if self.layer_number == 1:
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, args.mhc_rate, 1)

        # ------------------ Attn mHC Preprocess ------------------
        H_pre, H_post, H_res = self.attn_mhc.mapping(hidden_states, sinkhorn_knopp_batched)
        hidden_states, residual = self.attn_mhc.process(hidden_states, H_pre, H_res)
        # ---------------------------------------------------------

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        # For minicpm model
        if args.scale_depth is not None:
            attention_output, attention_bias = attention_output_with_bias
            attention_output = attention_output * (args.scale_depth / math.sqrt(args.num_layers))
            attention_output_with_bias = (attention_output, attention_bias)

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        # ------------------ Attn mHC Postprocess ------------------
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout, H_post
            )
        # ---------------------------------------------------------

        # 把pre_cross_attn_layernorm和cross_attention非identity的情况排除掉
        assert isinstance(self.pre_cross_attn_layernorm, IdentityOp) and isinstance(self.cross_attention, IdentityOp), \
            "mHCTransformerLayer only supports pure self-attention layers not for cross-attention op."

        # ------------------ MLP mHC Preprocess ------------------
        H_pre_mlp, H_post_mlp, H_res_mlp = self.mlp_mhc.mapping(hidden_states, sinkhorn_knopp_batched)
        hidden_states, residual = self.mlp_mhc.process(hidden_states, H_pre_mlp, H_res_mlp)
        # ---------------------------------------------------------

        # Optional Layer norm post the cross-attention.
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_norm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        return pre_mlp_layernorm_output, residual, context, H_post_mlp

    def _forward_mlp(self, pre_mlp_layernorm_output, residual, mHC_H_post):
        args = get_args()
        # MLP.
        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, pre_mlp_layernorm_output
            )
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        if args.scale_depth is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = mlp_output * (args.scale_depth / math.sqrt(args.num_layers))
            mlp_output_with_bias = (mlp_output, mlp_bias)

        # ------------------ MLP mHC Postprocess ------------------
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout, mHC_H_post
            )
        # ---------------------------------------------------------

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output


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

        transformer_layer_spec.module = mHCTransformerLayer
        transformer_layer_spec.submodules.self_attn_bda = get_bias_dropout_add_mHC
        transformer_layer_spec.submodules.mlp_bda = get_bias_dropout_add_mHC

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
    parser.add_argument('--mhc-rate', type=int, default=4,
                        help='The rate (number of hyper-connections) for manifold hyper-connection.')
    parser.add_argument('--mhc-max-sk-it', type=int, default=20,
                        help='The max number of Sinkhorn-Knopp iterations for manifold hyper-connection.')
    return parser


def main():
    print_rank_0(f"{'*'*30}\nPretraining gpt model (V2)...\n{'*'*30}")

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=_extra_args_provider,
    )


if __name__ == "__main__":
    main()
