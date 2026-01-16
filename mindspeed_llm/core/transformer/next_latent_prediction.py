# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, make_viewless_tensor


try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDelayedScaling,
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    import apex  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn('Apex is not installed. Falling back to Torch Norm')
    LNImpl = WrappedTorchNorm



def roll_tensor(tensor, shifts=-1, dims=-1):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(0)
    return rolled_tensor, rolled_tensor.sum()


class MTPLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, mtp_loss: torch.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            mtp_loss (torch.Tensor): The mtp loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = torch.ones_like(mtp_loss) * mtp_loss_backward_scale
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


class NextLatentPredLossLoggingHelper:
    """Helper class for logging Next Latent Prediction (NextLat) losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        step_index: int,
        num_steps: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the NextLat loss for logging.

        Args:
            loss (torch.Tensor): The loss tensor (scalar or 1D).
            step_index (int): Index of the prediction step (0 to num_steps-1).
            num_steps (int): Total number of prediction steps (horizon).
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss (e.g., sum).
            avg_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip logging if step_index is None.
        if step_index is None:
            return

        tracker = NextLatentPredLossLoggingHelper.tracker
        
        # Initialize the storage tensor if it doesn't exist
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_steps, device=loss.device)
            
        # Accumulate the detached loss
        # Note: We use step_index to index into the values tensor
        if step_index < num_steps:
             tracker["values"][step_index] += loss.detach()
        
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the stored losses in tracker."""
        tracker = NextLatentPredLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the NextLat losses across ranks."""
        tracker = NextLatentPredLossLoggingHelper.tracker
        if "values" not in tracker:
            return
            
        values = tracker["values"]
        
        # Reduce losses across ranks (e.g., across Data Parallel groups)
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
            
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    @staticmethod
    def track_nextlat_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """
        Track the Next Latent Prediction metrics for logging.
        
        Args:
            loss_scale (float): The scaling factor (e.g. 1.0 / micro_batch_size).
            iteration (int): Current iteration step.
            writer: TensorBoard writer.
            wandb_writer: WandB writer.
            total_loss_dict (dict): Dictionary to store the reduced losses.
        """
        # 1. Reduce across GPUs
        NextLatentPredLossLoggingHelper.reduce_loss_in_tracker()
        
        tracker = NextLatentPredLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        # 2. Scale the loss
        # The stored value is sum(loss), so we multiply by scale (usually 1/batch_size)
        nextlat_losses = tracker["values"] * loss_scale
        num_steps = nextlat_losses.shape[0]

        # 3. Log each step's loss
        for i in range(num_steps):
            # Naming convention: next_latent_1 loss, next_latent_2 loss...
            name = f"next_latent_{i+1} loss"
            loss = nextlat_losses[i]

            if total_loss_dict is not None:
                total_loss_dict[name] = loss
            
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        # 4. Clean up for the next iteration
        NextLatentPredLossLoggingHelper.clean_loss_in_tracker()


class NextLatentPredictionLayer(MegatronModule):
    """The implementation for Next Latent Prediction Layer
    """

    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__(config=config)

        self.layer_norm = nn.LayerNorm(
            normalized_shape=self.config.hidden_size*2,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_fc1 = ColumnParallelLinear(
            self.config.hidden_size*2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_fc2 = RowParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_fc3 = ColumnParallelLinear(
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=True,
            bias=self.config.add_bias_linear,
            skip_bias_add=False,
            is_expert=False,
        )

    def forward(
        self,
        decoder_input: Tensor,
        hidden_states: Tensor,
    ):
        """
        Perform the forward pass through the Next Latent Prediction layer.

        Args:
            hidden_states (Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            decoder_input (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
                At the (k - 1)-th MTP module, the i-th element of decoder input is
                the embedding of (i + K)-th tocken.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        fp8_context = nullcontext()

        with rng_context, fp8_context:
            _input = torch.cat((decoder_input, hidden_states), -1)
            _input = self.layer_norm(_input)

            _input = make_viewless_tensor(
                inp=_input, requires_grad=True, keep_graph=True
            )

            output, _ = self.linear_fc1(_input)
            output, _ = self.linear_fc2(F.gelu(output))
            output, _ = self.linear_fc3(F.gelu(output))

            output = all_gather_last_dim_from_tensor_parallel_region(output)
            output = hidden_states + output

        output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)

        return output


class NextLatentPredictionBlock(MegatronModule):
    """The implementation for Next Latent Prediction Block
    """

    def __init__(
        self, config: TransformerConfig,
    ):
        super().__init__(config=config)
        self.nextlp_layer = NextLatentPredictionLayer(config=config)
        self.nextlp_steps = config.next_latent_prediction_steps
        self.lambda_next_h = config.lambda_next_h
        self.lambda_kl = config.lambda_kl
        self.nextlp_loss_scaling_factor = config.nextlp_loss_scaling_factor

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[Tensor] = None,
        embedding=None,
        output_layer=None,
        output_weight: Optional[torch.Tensor] = None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP modules.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = torch.ones_like(input_ids)

        hidden_states_main_model = hidden_states  # [s, b, h]
        
        # Target state: continuously roll this to get h_{t+1}, h_{t+2} as Ground Truth
        target_hidden_states = hidden_states  # [s, b, h]

        # Target probs: continuously roll this to get P_{t+1}, P_{t+2} as Ground Truth
        output_weight = output_layer.weight.detach() if output_weight is None else output_weight.detach()
        with torch.no_grad():
            target_logits, _ = output_layer(
                target_hidden_states.detach(), weight=output_weight, 
                runtime_gather_output=runtime_gather_output
            )
            target_log_probs = F.log_softmax(target_logits, dim=-1)  # [s, b, v]

        for step_idx in range(self.nextlp_steps):
            # Calc logits for the current next latent predction steps
            input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)  # [b, s]

            # embedding
            decoder_input = embedding(input_ids=input_ids, position_ids=position_ids) # [s, b, h]
            
            # norm, mlp projection and residual connection
            hidden_states = self.nextlp_layer(  # [s, b, h]
                decoder_input=decoder_input,
                hidden_states=hidden_states,
            )

            # Calc loss for the current next latent prediction step.
            # Additional Loss: \lambda_{next-h} * L_{next-h} + \lambda_{KL} * L_{KL}
            loss_mask, num_tokens = roll_tensor(loss_mask, shifts=-1, dims=-1)              # [b, s]
            target_hidden_states, _ = roll_tensor(target_hidden_states, shifts=-1, dims=0)  # [s, b, h]
            target_log_probs, _ = roll_tensor(target_log_probs, shifts=-1, dims=0)          # [s, b, v]

            reg_loss = F.smooth_l1_loss(hidden_states, target_hidden_states.detach(), reduction='none').mean(-1)
            reg_loss = reg_loss.transpose(0, 1).contiguous() # [s, b] -> [b, s] 
            reg_loss = reg_loss * loss_mask

            pred_logits, _ = output_layer(
                hidden_states, weight=output_weight, 
                runtime_gather_output=runtime_gather_output
            )
            pred_log_probs = F.log_softmax(pred_logits, dim=-1)
            kl_loss = F.kl_div(pred_log_probs, target_log_probs, log_target=True, reduction='none').sum(-1)
            kl_loss = kl_loss.transpose(0, 1).contiguous()  # [s, b] -> [b, s]
            kl_loss = kl_loss * loss_mask

            nextlp_step_loss = (self.lambda_next_h * reg_loss) + (self.lambda_kl * kl_loss)

            if self.training:
                NextLatentPredLossLoggingHelper.save_loss_to_tracker(
                    loss=torch.sum(nextlp_step_loss) / num_tokens,
                    step_index=step_idx,
                    num_steps=self.nextlp_steps,
                    reduce_group=parallel_state.get_data_parallel_group(),
                    avg_group=parallel_state.get_data_parallel_group(),
                )

            nextlp_loss_scale = self.nextlp_loss_scaling_factor / self.config.next_latent_prediction_steps
            if self.config.calculate_per_token_loss:
                hidden_states_main_model = MTPLossAutoScaler.apply(
                    hidden_states_main_model, nextlp_loss_scale * nextlp_step_loss
                )
            else:
                hidden_states_main_model = MTPLossAutoScaler.apply(
                    hidden_states_main_model, nextlp_loss_scale * nextlp_step_loss / num_tokens
                )

        return hidden_states_main_model

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the multi token prediction module.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the multi
            token prediction module.
        """
        # Get the base class state_dict (usually empty or common metadata)
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # Define the prefix for the sub-module
        layer_prefix = f'{prefix}nextlp_layer.'

        # Recursively call the sharded_state_dict of the sub-module (nextlp_layer).
        # Here, unlike MTP, we do not need to calculate get_mtp_layer_offset,
        # because the NextLat Layer is a simple MLP and does not participate in
        # the layer-wise partitioning of Pipeline Parallelism.
        layer_sharded_state_dict = self.nextlp_layer.sharded_state_dict(
            layer_prefix, sharded_offsets, metadata
        )
        
        sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict