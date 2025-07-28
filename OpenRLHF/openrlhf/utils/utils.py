from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def get_strategy(args):
    """
    Get the strategy for distributed training. 
    返回一个 用于分布式训练的策略对象，
    具体是 openrlhf.utils.deepspeed 模块中的 DeepspeedStrategy 实例，
    它封装了 DeepSpeed 的配置参数，并为 RLHF 模型训练做好准备。
    """
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),                      # 设置随机种子
        full_determinism=getattr(args, "full_determinism", False),  # 是否启用完全确定性训练
        max_norm=getattr(args, "max_norm", 1.0),             # 梯度最大范数（用于裁剪）
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),  # 单个 GPU 上的 batch size
        train_batch_size=getattr(args, "train_batch_size", 128),            # 全局 batch size（所有 GPU 总和）
        zero_stage=args.zero_stage,                          # DeepSpeed ZeRO 优化阶段（0~3）
        bf16=getattr(args, "bf16", True),                    # 是否启用 bfloat16 精度
        args=args,                                           # 传入所有命令行参数，用于后续引用
    )
    """
    返回的是一个 DeepspeedStrategy 对象，你可以理解为一个「训练策略控制器」，它统一管理：
    - 分布式配置（如 ZeRO stage）
    - 混合精度（bf16）
    - 梯度裁剪（max norm）
    - 批大小（micro vs. global）
    - 随机种子、是否完全确定性（determinism）
    - 原始命令行参数（保存在 strategy.args 中）
    """
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch
