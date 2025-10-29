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

from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DatasetBuilder:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        shuffle_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_seed = shuffle_seed

    def build_dataset(self, datapath: str, num_proc: int = 8) -> Dataset:
        try:
            # Load and shuffle dataset
            ds = load_dataset("json", data_files=datapath)
            ds = ds["train"].shuffle(seed=self.shuffle_seed)

            # Store original columns for removal
            original_columns = ds.column_names

            # Apply preprocessing
            processed_ds = ds.map(
                self._preprocess_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns,
                load_from_cache_file=False,
                desc="Processing conversations",
            )

            # Filter out None results
            processed_ds = processed_ds.filter(lambda x: x["input_ids"] is not None)
            processed_ds.set_format(type="torch")

            return processed_ds

        except Exception as e:
            raise RuntimeError(f"Dataset building failed for {datapath}") from e

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        new_examples = {"input_ids": [], "attention_mask": [], "loss_mask": []}

        for i in range(len(examples["id"])):
            try:
                processed_example = self._process_single_conversation(
                    examples["conversations"][i]
                )

                if processed_example is not None:
                    for key, value in processed_example.items():
                        if key in new_examples:
                            new_examples[key].append(value)

            except Exception as e:
                # TODO: rank0 print
                print(f"Error processing example: {e}")
                # Add None placeholders to maintain batch consistency
                for key in new_examples:
                    new_examples[key].append(None)

        return new_examples

    def _process_single_conversation(
        self, conversation_data: List[Dict]
    ) -> Optional[Dict]:
        if not conversation_data or not isinstance(conversation_data, list):
            return None

        try:
            # Build messages with system prompt
            messages = self._build_messages(conversation_data)
            if not messages:
                return None

            input_ids_list = []
            loss_mask_list = []

            for message in messages:
                message_tokens = self.tokenizer.apply_chat_template(
                    [message],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                ).squeeze(0)

                # Determine the loss mask based on the role
                if message["role"] in ["system", "user"]:
                    mask = torch.zeros_like(message_tokens)
                else:  # assistant
                    mask = torch.ones_like(message_tokens)

                input_ids_list.append(message_tokens)
                loss_mask_list.append(mask)

            input_ids = torch.cat(input_ids_list, dim=0)
            loss_mask = torch.cat(loss_mask_list, dim=0)

            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]

            attention_mask = torch.ones_like(input_ids)

            return {
                "input_ids": input_ids[None, :],
                "attention_mask": attention_mask[None, :],
                "loss_mask": loss_mask[None, :],
            }

        except Exception as e:
            # TODO: rank0 print
            print(f"Error processing conversation: {e}")
            return None

    def _build_messages(self, source: List[Dict]) -> List[Dict]:
        # System message
        messages = [{"role": "system", "content": self._get_system_prompt()}]

        # Role mapping
        role_mapping = {"human": "user", "gpt": "assistant"}
        expected_roles = ["user", "assistant"]

        # Ensure conversation starts with user
        if source[0]["from"] != "human":
            source = source[1:]

        # Filter and validate conversation turns
        valid_turns = []
        for turn in source:
            # TODO: 数据集改成openai格式
            if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
                continue

            role = role_mapping.get(turn["from"])
            if role and turn["value"].strip():
                valid_turns.append({"role": role, "content": turn["value"].strip()})

        # Validate alternating pattern
        for i, turn in enumerate(valid_turns):
            expected_role = expected_roles[i % 2]
            if turn["role"] != expected_role:
                break
            messages.append(turn)

        return messages if len(messages) > 1 else []

    def _get_system_prompt(self) -> str:
        """Get the system prompt for conversations."""
        return (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, racist, "
            "sexist, toxic, dangerous, or illegal content. Please ensure that "
            "your responses are socially unbiased and positive in nature.\n\n"
            "If a question does not make any sense, or is not factually coherent, "
            "explain why instead of answering something not correct. If you don't "
            "know the answer to a question, please don't share false information."
        )


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


class DatasetManager:
    """
    Simplified DatasetManager for train_eagle3_online.py.

    This manager is designed to work with DataArguments from train_eagle3_online.py
    and provides a simple interface to create train and eval datasets.
    """

    def __init__(
        self,
        data_args,
        tokenizer: AutoTokenizer,
        model_max_length: int = 2048,
    ):
        """
        Initialize DatasetManager with DataArguments.

        Args:
            data_args: DataArguments object from train_eagle3_online.py
            tokenizer: Tokenizer for the model
            model_max_length: Maximum sequence length
        """
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        # Create dataset builder
        self.dataset_builder = DatasetBuilder(
            tokenizer=tokenizer,
            max_length=model_max_length,
            shuffle_seed=data_args.shuffle_seed,
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
        if self.data_args.preprocessing_num_workers is not None:
            num_proc = self.data_args.preprocessing_num_workers

        # Create train dataset
        train_dataset = self.dataset_builder.build_dataset(
            self.data_args.train_data_path, num_proc=num_proc
        )

        # Create eval dataset if path is provided
        eval_dataset = None
        if self.data_args.eval_data_path is not None:
            eval_dataset = self.dataset_builder.build_dataset(
                self.data_args.eval_data_path, num_proc=num_proc
            )

        return train_dataset, eval_dataset
