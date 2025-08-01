from typing import Optional

import torch
import torch.distributed as dist
# 分布式通信（例如 all-gather 等），在后面用于处理 packing 后的数据聚合。
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
# BitsAndBytesConfig：用于配置 4-bit 量化（配合 bitsandbytes 实现低精度加载）。
# HfDeepSpeedConfig：非 Trainer 情境下与 DeepSpeed 配合的配置封装（避免全局副作用）。
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor
from .utils import compute_entropy, log_probs_from_logits
# ring_attn_utils.gather_and_pad_tensor / unpad_and_slice_tensor：
#   用于 “packed samples” 场景下把变长 / padding 处理后的张量展开 / 恢复，配合分布式 group。



class Actor(nn.Module):
    """
    Base class for Actor models in reinforcement learning.

    This class serves as a foundation for implementing various actor models, which are responsible for 【selecting actions】 based on the policy learned from the environment.

    Args:
        pretrain_or_model (nn.Module): A pretrained model or a new model instance to be used as the actor.
        use_flash_attention_2 (bool, optional): Whether to utilize Flash Attention 2.0 for improved performance. Defaults to False.
        bf16 (bool, optional): Enable bfloat16 precision for model computations. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        target_modules (list, optional): List of target modules for applying LoRA. Defaults to None.
        ds_config (dict, optional): Configuration for DeepSpeed, enabling model partitioning across multiple GPUs. Defaults to None.
        device_map (dict, optional): Device mapping for loading the model onto specific devices. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        temperature (float, optional): Temperature for action selection. Defaults to 1.0.
        use_liger_kernel (bool, optional): Whether to use Liger Kernel for the model. Defaults to False.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        temperature=1.0,
        use_liger_kernel=False,
        **kwargs,
    ) -> None:
        
        # packing samples（batch 内变长/拼接优化）。
        # 可选使用自定义 kernel（比如 liger_kernel，用于更快/更特化的模型实现）。
        # MoE 支持（后面通过 config 判断）
        # 

        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            # --- attention 实现 --- #
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            # --- DeepSpeed 配置 --- #
            # Note: dschf is defined in function scope to avoid global effects
            # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            
            # 仅在 ZeRO Stage 3 时包装成 HfDeepSpeedConfig。
            # 这是为了与 Transformers 直接集成 DeepSpeed（非 Trainer 模式），避免全局副作用。
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            # --- 4-bit 量化配置 --- #
            if load_in_4bit:
                assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            # ---  模型类选择（Liger Kernel vs 标准） --- #
            #  LinkedIn 开发的 Liger Kernel
            #  Liger Kernel 是一套针对大语言模型（LLM）训练/微调的 高效 Triton 内核集合，
            #  目的是在不改动模型结构大幅提升吞吐和内存效率。
            #  它通过融合常见操作、定制化低级 kernel 实现、以及对关键组件（比如 RMSNorm、RoPE、SwiGLU、CrossEntropy、FusedLinearCrossEntropy 等）的高度优化，达到典型效果：
            #   训练吞吐提升约 20%，
            #   GPU 显存使用下降 ~60%。
            #   它和 Flash Attention、PyTorch FSDP、DeepSpeed 原生兼容，并且对常见对齐/蒸馏/RLHF 类 loss（如 DPO, CPO, ORPO 等）提供优化。
            if use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM

                model_class = AutoLigerKernelForCausalLM
            else:
                model_class = AutoModelForCausalLM

            self.model = model_class.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                # enable_input_require_grads() 是为了符合特定 issue 的 workaround（让输入梯度传播正常）。
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                # 【引入LoRA后的混合精度设置】
                # 之后有一段为 4-bit 情况下做精度修正（norm 还是用 float32、LoRA layer 转为 bf16，
                # 某些嵌入 / head 也调整）——这是为了兼顾稳定性和效率。
                # 
                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - balancing loss
            # ？？？如果模型是 MoE 类型（会有 router logits），
            # 开启 router logits 输出用于后续 loss/平衡等处理
            # （通常 MoE 需要 balancing loss 依赖这些 logits）。
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logits as True")
                self.model.config.output_router_logits = True

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            # 避免生成时使用缓存（历史 key/value），
            # 因为某些训练/梯度场景下与 LoRA / packing 组合会出问题。
            # Issue 提示建议用 generate(use_cache=True) 的替代，但这里强制训练时关闭。
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            # DEF：packing 技术是一种通过优化数据组织方式提升训练效率的核心方法。
            #      其核心思想是将多个短序列合并为一个长序列，最大限度减少填充（Padding）带来的计算浪费
            # 记录是否启用了 sample packing，用于后续 forward 分支逻辑。
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model


            """
            【以下是Packing操作需要在forward中做出的设计】
            Packing 本质上是结构变换+压缩（去掉 padding、重排、拼接），它带来效率但破坏了原始序列与预测目标之间的简单一一对应。
            为了恢复这种对应、在 distributed/ring-attn 环境下保持准确、还能正确计算 policy 所需的 log-prob/entropy/action mask，
            代码必须显式解包（unpad_and_slice_tensor）、再聚合（gather_and_pad_tensor）、并在 shift、mask、temperature、输出格式上做细致对齐。
            这个“复杂”其实是为了在不牺牲精度的前提下提取 packing 的性能收益所必需的折衷。
            1. position id和roll（目标token）如何对齐
            packing 后原始的 padding 被裁掉、各样本拼接、内部可能做了重排序（unpad_and_slice_tensor 里一般会做 slice 与 reindex），原始的 shift/位置关系失效了。所以需要：
            - unpad_and_slice_tensor 处理去 padding、生成新的 rolled_sequences（目标）和 position_ids（可能在内部已经正确构造），
            - 之后再用 gather_and_pad_tensor 把 model 输出恢复到原来 batch+seqlen 结构（用于 loss、mask 应用、entropy 之类）。
            
            2. attention mask 处理
            packing 之后原来的 attention_mask 不能直接送进去（因为 token 被 compact 了），so：
            - 在 packing 路径里把 attention_mask 通过 unpad_and_slice_tensor 变成 model 能接受的形式（通常不再显式传 attention_mask，由内部处理隐式表示），
            - 非 packing 时才按传统方式构造 position_ids 与 attention_mask。
            因此要分支处理：一条是“pack 过的、internal attention 处理”，另一条是“普通 padded sequence”。
            
            3. 分布式 / ring-attn 兼容性
            出现 ring_attn_group、gather_and_pad_tensor 这些，是因为在分布式、特别是 ring-attention 或自定义通信 (比如把多个 GPU 上的 packed slice 合并/还原）时：
            - packing 后的 tensor 在各个进程/分组里长度不一致，必须通过 gather_and_pad_tensor 做 all-gather + 恢复到一致 shape（否则后续计算/对齐（如 entropy、log-prob）会错）。
            - unpad_and_slice_tensor 通常也会返回 indices/pad_len，用于后面的 “解包还原” 操作。

            4. log-prob / entropy 的对齐与截断
            由于语言模型预测是“下一个 token”，所以常见要做 shift。pack 后这个 shift 也要和原始样本对齐：
            - rolled_sequences 已经在 packing 过程中做了正确的 shift（如果直接 roll 会把不同样本互相污染）。
            - 得到的 log-prob 必须再通过 gather_and_pad_tensor 还原到 [batch, seqlen] 形状，然后 [:, :-1] 去除因为 shift 产生的多余末尾对齐。
            - entropy 同理，也要做 shape 恢复和截断。
            如果不做这些精细处理（比如直接 roll 后算 log-prob，再 blind 乘 mask），action 的 log-prob 取错位置、entropy 也无法对应原始 token，会导致 policy gradient 计算出错、reward attribution 失真。
            
            
            """

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        allgather_logits=False,
        return_logprobs=False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
        return_entropy=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        # 这里的actor的action是指模型的输出 logits

        batch, seqlen = sequences.size()
        foward_attention_mask = attention_mask

        # --- packing samples分支【对attention逻辑进行衡量】 --- #
        if self.packing_samples:
            # unpad_and_slice_tensor：处理变长样本的 packing（去除 padding、重组、为 ring-attn 等做 slice），返回的 rolled_sequences 是为了后面 log-prob 计算 shift 的版本。
            # 注意 forward_attention_mask 被置为 None，因为 packing 之后 attention 的逻辑由 utils 内部处理。
            
            sequences, position_ids, rolled_sequences, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                sequences, attention_mask, ring_attn_group
            )
            foward_attention_mask = None
        # --- 非 packing 分支【用torch的处理逻辑就可以】 --- #
        # rolled_sequences：表示下一token 的“目标”用来计算 log-prob，类似语言模型 teacher-forcing 中的 shift。
        # position_ids 基于 attention mask 计算（累加），再把 padding 位置填成 1（避免非法）。
        #   这个构造是为了解决某些模型在 variable-length 输入下 position embedding 的对齐问题。参考 issue 217 的实践细节。
        else:
            # 正常不 packing 时，用 torch.roll(sequences, -1) 得到下一个 token 作为目标
            # 位置 id 用累加 attention_mask 构造可以直接用。
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        
        # --- forward过程 --- #
        # 这里logits 强制转为 float32（避免后续在 bf16 / 4bit 场景下精度过低影响 log-prob 计算）。
        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids)
        # https://github.com/OpenRLHF/OpenRLHF/pull/634
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            entropy = compute_entropy(output["logits"])
            # packing 时需要重新 gather/恢复原始 shape。
            if self.packing_samples:
                entropy = gather_and_pad_tensor(entropy, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            # 最终把 entropy 绑定到 output 上，并去除最后一个 token（因为 rolled target 也会 shift 一位）。
            setattr(output, "entropy", entropy[:, :-1])
        
        # --- 返回 1: 用户只需要output --- #
        # packing仍然需要做处理：allgather_logits + packing 时会做额外 gather 恢复 logits。
        return_action_log_probs = action_mask is not None
        if not return_action_log_probs and not return_logprobs:
            assert return_output
            if allgather_logits and self.packing_samples:
                output["logits"] = gather_and_pad_tensor(
                    output["logits"], ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )
            return output

        # --- 返回 2: 用户需要 log probs --- #
        # 把 logits 和目标 sequence（shift 后的 rolled_sequences）输入，计算每个 token 的 log-prob，temperature 控制 flat/sharp。
        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)

        # 如果 packing，则需要再 gather 回原始 shape：
        if self.packing_samples:
            log_probs = gather_and_pad_tensor(log_probs, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

        # 丢掉最后一位（因为被 shift 了）：
        log_probs = log_probs[:, :-1]
        if not return_action_log_probs and return_logprobs:
            return (log_probs, output) if return_output else log_probs

        # --- 返回 3: 用户需要 action log probs --- #
        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()

        return (action_log_probs, output) if return_output else action_log_probs

    # 工具函数
    # 前两个包装底层模型的 gradient checkpointing 控制：用于训练时节省显存（通过在前向保存更少中间、反向再 recompute）
    # 额外 param use_reentrant 控制 checkpoint 的实现细节（兼容性/性能）。
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    # print_trainable_parameters：方便打印哪些参数是可训练的（对 LoRA 或 freeze 策略调试非常有用）。
    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
