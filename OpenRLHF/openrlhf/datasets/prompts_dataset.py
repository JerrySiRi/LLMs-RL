
# TODO 处理prompts

from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label

"""
继承自torch.utils.data.Dataset的PromptDataset类，用于处理PPO模型的数据集。
1. 该类在初始化时接收数据集、【分词器，负责把原始的prompt经由chat template转化成prompt】、策略和可选的输入模板。
2. 它会预处理数据集中的每个数据项，生成相应的提示和标签，并存储数据源信息。
3. 通过重载__len__和__getitem__方法，可以获取数据集的长度和指定索引的数据项。
4. 该类还支持【应用聊天模板来格式化输入数据】。
"""

# 有当类继承了 torch.utils.data.Dataset 并实现了 __len__ 和 __getitem__ 这两个方法，PyTorch 才能识别它是个可用的数据集。
# 否则，你就不能把它传给 torch.utils.data.DataLoader 了。

#
class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        
        """
        from abc import ABC, abstractmethod
        class Dataset(ABC):
            def __init__(self):
                pass  # 默认什么也不做

            @abstractmethod
            def __getitem__(self, index):
                raise NotImplementedError

            @abstractmethod
            def __len__(self):
                raise NotImplementedError

        Dataset 继承自 ABC（抽象基类），表示这是一个“接口类”。
        __init__() 是空的，不做任何初始化工作。
        __getitem__() 和 __len__() 被标注为 @abstractmethod【表示你必须在子类中实现它们，否则不能实例化】

        """

        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        # desc在进度条前面显示的文字说明，用于提示当前正在处理的任务是“Preprocessing data”。
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        # 必须要实现
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        # 必须要实现
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
