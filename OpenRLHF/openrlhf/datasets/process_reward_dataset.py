# 处理PRM的dataset

from typing import Callable
import torch
from torch.utils.data import Dataset
from openrlhf.utils.utils import convert_token_to_id, zero_pad_sequences


class ProcessRewardDataset(Dataset):
    """
    Dataset for process reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)
        self.placeholder_token = getattr(self.strategy.args, "placeholder_token", None)
        self.reward_tokens = getattr(self.strategy.args, "reward_tokens", None)

        # 把placeholder_token转换为token id
        self.placeholder_token_id = convert_token_to_id(self.placeholder_token, self.tokenizer)

        # Store the processed data in class attributes
        # 从dataset中获取输入和标签
        self.inputs = dataset[self.input_key]
        self.labels = dataset[self.label_key]

    def __len__(self):
        # 必须要实现
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        # 必须要实现
        """
        将第 idx 个输入字符串进行分词，按规则将标签（字符串或浮点数）编码成 tensor，并与输入中的占位符位置对齐，
        【也即把process reward放到每一步中去】
        返回模型可用的 (input_ids, attention_mask, labels)。
        """
        # ---- Tokenization ---- #
        input_token = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt", # 返回的是 PyTorch Tensor
            add_special_tokens=False,
        )
        # 得到的 input_token 是字典：包含 input_ids 和 attention_mask
        input_ids = input_token["input_ids"]

        # --- 准备label_tensor --- #
        label_values = self.labels[idx]
        assert isinstance(label_values, list), "labels should be a list of strings or numbers"
        # 检查 label_values 中的每个元素是否都是字符串或数字（两种情况）
        if isinstance(label_values[0], str):
            label_tokens = []
            for label in label_values:
                assert (
                    self.reward_tokens is None or label in self.reward_tokens
                ), f"label should be in reward tokens {self.reward_tokens}, got {label}"
                label_tokens.append(convert_token_to_id(label, self.tokenizer))

            # label_tokens is list of token id (for '+', '-', etc)
            label_tensor = torch.tensor(label_tokens, dtype=input_ids.dtype)
        else:
            # label_values is list of float numbers (for reward values)
            label_tensor = torch.tensor(label_values, dtype=torch.float)
        

        # ----- 对其标签和输入中的占位符 ------ #
        # TODO 将标签只填在 input_ids 中那些等于占位符 token ID 的位置，其它位置标记为 -100（PyTorch 中 -100 表示忽略）。
        
        # Motivation: inputs_ids maybe truncated to self.max_length, where placeholder_tokens at the end may be removed.
        # We should also truncate the labels to match the length of input_ids
        
        # Step 1: Create a mask for placeholder token positions
        mask = input_ids == self.placeholder_token_id
        
        # Step 2: Ensure that label_tensor is truncated along the last dimension
        # Find the length of the last dimension of the mask
        num_placeholders = mask.sum(dim=-1)
        # Truncate label_tensor along the last dimension to match num_placeholders
        truncated_labels = label_tensor[..., : num_placeholders.max()]
        
        # Step 3: Update labels at placeholder token positions
        labels = torch.full_like(input_ids, -100)
        labels[mask] = truncated_labels

        return (
            input_ids,
            input_token["attention_mask"],
            labels,
        )

    def collate_fn(self, item_list):
        """
        将多个 (input_ids, attention_mask, labels) 三元组进行 padding，组成一个统一长度的 batch。
        """
        input_ids = []
        input_masks = []
        label_ids = []
        for input_id, input_mask, label_id in item_list:
            input_ids.append(input_id)
            input_masks.append(input_mask)
            label_ids.append(label_id)

        padding_side = "right"
        input_ids = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks, side=padding_side)
        label_ids = zero_pad_sequences(label_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        return input_ids, input_masks, label_ids
