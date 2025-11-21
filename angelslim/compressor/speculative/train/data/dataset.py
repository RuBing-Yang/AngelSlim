# Copyright 2025 Tencent Inc. All Rights Reserved.
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

from typing import Optional, Tuple, Union

from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from angelslim.utils import rank0_print

from .chat_templates import ChatTemplateType, string_to_chat_template_type
from .dataset_builder import DatasetBuilderFactory


class DatasetManager:
    """
    Simplified DatasetManager for train_eagle3_online.py.

    This manager is designed to work with DataArguments from train_eagle3_online.py
    and provides a simple interface to create train and eval datasets.
    """

    def __init__(
        self,
        data_args,
        modal_type: str,
        tokenizer: Union[AutoTokenizer, AutoProcessor],
        model_max_length: int = 2048,
        chat_template_type: Optional[Union[str, ChatTemplateType]] = None,
        display: bool = False,
    ):
        """
        Initialize DatasetManager with DataArguments.

        Args:
            data_args: DataArguments object from train_eagle3_online.py
            tokenizer: Tokenizer for the model
            model_max_length: Maximum sequence length
            chat_template_type: Chat template type. Can be:
                - ChatTemplateType enum value (e.g., ChatTemplateType.QWEN3)
                - String (e.g., "llama", "qwen")
                - None (will default to LLAMA)
            display: Whether to display loss mask visualization for the first sample
        """
        self.data_args = data_args
        self.modal_type = modal_type
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.display = display

        # Convert chat_template_type to ChatTemplateType enum
        if chat_template_type is None:
            # Default to QWEN3
            chat_template_type = ChatTemplateType.QWEN3
        elif isinstance(chat_template_type, str):
            # Convert string to enum
            chat_template_type = string_to_chat_template_type(chat_template_type)

        rank0_print(f"modal_type={self.modal_type}")

        # Create dataset builder
        self.dataset_builder = DatasetBuilderFactory.create(
            modal_type,
            tokenizer=tokenizer,
            max_length=model_max_length,
            shuffle_seed=data_args.shuffle_seed,
            chat_template_type=chat_template_type,
            display=display,
        )

    def create_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Create train and eval datasets based on DataArguments.

        Returns:
            Tuple of (train_dataset, eval_dataset)
            eval_dataset will be None if eval_data_path is not provided
        """
        # Determine number of processes
        num_proc = self.data_args.num_proc
        sample_num = self.data_args.sample_num
        if self.display:
            num_proc = None

        # Create train dataset
        train_dataset = self.dataset_builder.build_dataset(
            self.data_args.train_data_path, num_proc=num_proc, sample_num=sample_num
        )

        # Create eval dataset if path is provided
        eval_dataset = None
        if self.data_args.eval_data_path is not None:
            eval_dataset = self.dataset_builder.build_dataset(
                self.data_args.eval_data_path, num_proc=num_proc, sample_num=sample_num
            )

        data_collator = self.dataset_builder.get_data_collator()

        rank0_print(f"Train dataset size: {len(train_dataset)} samples")

        return train_dataset, eval_dataset, data_collator
