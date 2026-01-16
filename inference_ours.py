# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

from mindspeed_llm import megatron_adaptor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from mindspeed_llm.tasks.inference.infer_base import task_factory
from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM

import os
import copy
import torch
from megatron.training.checkpointing import load_checkpoint
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.training.utils import unwrap_model
from megatron.core.rerun_state_machine import get_rerun_state_machine

# Global teacher handles reused across calls
global_teacher1 = None
global_teacher2 = None


def _ensure_teachers_initialized():
    """Load frozen stage1 (text->hidden) and stage2 (hidden->hidden) teachers."""
    global global_teacher1, global_teacher2
    if global_teacher1 is not None and global_teacher2 is not None:
        return

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

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

    def _load_teacher(load_arg_name, pre_process, post_process, config):
        teacher = GPTModel(
            config=config,
            transformer_layer_spec=teacher_layer_spec,
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
        )
        teacher.cuda(torch.cuda.current_device())
        if not hasattr(args, load_arg_name) or getattr(args, load_arg_name) is None:
            raise ValueError(f"--{load_arg_name.replace('_','-')} must be specified")
        args_snapshot = copy.deepcopy(args.__dict__)
        rerun_snapshot = None
        try:
            rerun_snapshot = get_rerun_state_machine().state_dict(data_iterator=None, ckpt_format=args.ckpt_format)
        except Exception:
            rerun_snapshot = None
        orig_flags = (
            getattr(args, 'no_load_rng', False),
            getattr(args, 'no_load_optim', False),
            getattr(args, 'finetune', False),
        )
        args.no_load_rng = True
        args.no_load_optim = True
        args.finetune = True

        ddp_like_teacher = [teacher]
        assert os.path.exists(getattr(args, load_arg_name)), f"teacher checkpoint path {getattr(args, load_arg_name)} does not exist!"
        _ = load_checkpoint(ddp_like_teacher, None, None, load_arg=load_arg_name, strict=False, allow_ckpt=True)
        teacher = unwrap_model(ddp_like_teacher)[0]

        # Restore flags
        for k, v in args_snapshot.items():
            setattr(args, k, v)
        if rerun_snapshot is not None:
            try:
                get_rerun_state_machine().load_state_dict(rerun_snapshot)
            except Exception:
                pass
        args.no_load_rng, args.no_load_optim, args.finetune = orig_flags

        # Freeze
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return teacher

    # Build separate configs per teacher to allow external layer control
    t1_config = copy.deepcopy(teacher_config)
    t2_config = copy.deepcopy(teacher_config)
    if hasattr(args, 'teacher1_num_layers') and args.teacher1_num_layers is not None:
        setattr(t1_config, 'num_layers', args.teacher1_num_layers)
    if hasattr(args, 'teacher2_num_layers') and args.teacher2_num_layers is not None:
        setattr(t2_config, 'num_layers', args.teacher2_num_layers)

    if global_teacher1 is None:
        global_teacher1 = _load_teacher('teacher1_load', pre_process=True, post_process=False, config=t1_config)
        print_rank_0("Loaded teacher1 for stage3 inference.")
    if global_teacher2 is None:
        global_teacher2 = _load_teacher('teacher2_load', pre_process=True, post_process=False, config=t2_config)
        print_rank_0("Loaded teacher2 for stage3 inference.")


def _forward(self, tokens, position_ids, attention_mask):
    """Forward step that routes tokens through teacher1/teacher2 before student.

    Overrides `_forward` to compute student logits from teacher-computed hidden states.
    """
    _ensure_teachers_initialized()
    args = get_args()
    t1 = global_teacher1
    t2 = global_teacher2

    with torch.no_grad():
        h1 = t1(tokens, position_ids, attention_mask)
        if isinstance(h1, tuple):
            h1 = h1[0]
        h1 = h1.detach()
        h2 = t2(
            decoder_input=h1,
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        if isinstance(h2, tuple):
            h2 = h2[0]
        h2 = h2.detach()

        teacher_embedding = None
        if hasattr(args, 'use_teacher_embedding_input') and args.use_teacher_embedding_input:
            teacher_embedding = t1.embedding(input_ids=tokens, position_ids=position_ids).detach()
            scale = getattr(args, 'teacher_embedding_scale', 1.0)
            if scale != 1.0:
                teacher_embedding = teacher_embedding * scale

        decoder_input = h2 if teacher_embedding is None else (h2 + teacher_embedding)

    # Call student with decoder_input path
    return self.model(
        decoder_input=decoder_input,
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        inference_context=self.inference_context,
    )


def model_provider(pre_process=True, post_process=True) -> Union[GPTModelInfer, GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModelInfer, GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.sequence_parallel and args.use_kv_cache:
        raise AssertionError('Use_kv_cache can not be true in sequence_parallel mode.')

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    else:
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        model = megatron.legacy.model.GPTModel(
            config,
            parallel_output=True if args.sequence_parallel else False,
            pre_process=pre_process,
            post_process=post_process
        )

    return model


def _extra_args_provider(parser):
    parser.add_argument('--teacher1-load', type=str, default=None,
                        help='Checkpoint path for stage1 teacher (text->hidden).')
    parser.add_argument('--teacher2-load', type=str, default=None,
                        help='Checkpoint path for stage2 teacher (hidden->hidden).')
    parser.add_argument('--teacher1-num-layers', type=int, default=None,
                        help='Number of transformer layers in stage1 teacher; overrides config for teacher1.')
    parser.add_argument('--teacher2-num-layers', type=int, default=None,
                        help='Number of transformer layers in stage2 teacher; overrides config for teacher2.')
    return parser


def main():
    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True},
        extra_args_provider=_extra_args_provider
    )

    args = get_args()

    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    # Swap ForwardStep to stage3 variant so generation uses teacher->student pipeline
    ForwardStep._forward = _forward

    task_factory(args, model)


if __name__ == "__main__":
    main()