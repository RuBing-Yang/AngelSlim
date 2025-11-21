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

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseBackend(ABC):
    """Base class for model backends"""

    def __init__(self, model_path: str, modal_type: str = "LLM", **kwargs):
        self.model_path = model_path
        self.modal_type = modal_type
        self.kwargs = kwargs
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the backend model"""
        pass

    @abstractmethod
    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get hidden states and logits from model"""
        pass


class VLMForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        inputs_embeds = None
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            inputs_embeds = kwargs["inputs_embeds"]
        elif len(args) > 2 and args[2] is not None:
            inputs_embeds = args[2]

        outputs = self.model.forward(*args, **kwargs)
        return outputs, inputs_embeds


class TransformersBackend(BaseBackend):
    """HuggingFace Transformers backend"""

    def load_model(self):

        default_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        default_kwargs.update(self.kwargs)

        if self.modal_type == "LLM":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_class = AutoModelForCausalLM
            tokenize_class = AutoTokenizer
        elif self.modal_type == "VLM":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_class = AutoModelForImageTextToText
            tokenize_class = AutoProcessor

        self.model = model_class.from_pretrained(self.model_path, **default_kwargs)

        if self.modal_type == "VLM":
            self.model = VLMForwardWrapper(self.model)

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.tokenizer = tokenize_class.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        inputs_embeds = None
        with torch.no_grad():
            if self.modal_type == "VLM":
                outputs, inputs_embeds = self.model(
                    input_ids,
                    attention_mask,
                    output_hidden_states=True,
                    output_logits=True,
                )
            else:
                outputs = self.model(
                    input_ids,
                    attention_mask,
                    output_hidden_states=True,
                    output_logits=True,
                )

        aux_hidden_states_layer_ids = kwargs.get("aux_hidden_states_layer_ids", None)
        if aux_hidden_states_layer_ids is None:
            out_hidden_nums = len(outputs.hidden_states)
            aux_hidden_states_layer_ids = [
                1,
                out_hidden_nums // 2 - 1,
                out_hidden_nums - 4,
            ]

        embed_offset = 1
        low_hidden_states = outputs.hidden_states[
            aux_hidden_states_layer_ids[0] + embed_offset
        ]
        mid_hidden_states = outputs.hidden_states[
            aux_hidden_states_layer_ids[1] + embed_offset
        ]
        high_hidden_states = outputs.hidden_states[
            aux_hidden_states_layer_ids[2] + embed_offset
        ]
        hidden_states = torch.cat(
            [low_hidden_states, mid_hidden_states, high_hidden_states], dim=-1
        )

        target = outputs.logits
        device = input_ids.device
        if inputs_embeds is not None:
            return hidden_states, target.to(device), inputs_embeds.to(device)
        return hidden_states, target.to(device), None


class TargetModelWrapper:
    """
    Target model wrapper for Eagle3 training.

    Supports three backends:
    - hf: HuggingFace Transformers AutoModelForCausalLM
    """

    BACKENDS = {
        "hf": TransformersBackend,
    }

    def __init__(self, backend: str, model_path: str, **kwargs):
        """
        Initialize TargetModel with specified backend

        Args:
            backend: One of ["hf"]
            model_path: Path to model
            **kwargs: Additional arguments for backend initialization
        """
        if backend not in self.BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Available backends: {list(self.BACKENDS.keys())}"
            )

        self.backend_name = backend
        self.backend = self.BACKENDS[backend](model_path, **kwargs)
        self.backend.load_model()

    def get_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get hidden states and logits from target model

        Args:
            input_ids: Input token ids, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]

        Returns:
            Tuple of (hidden_states, logits)
            - hidden_states: shape [batch_size, seq_len, hidden_size]
            - logits: shape [batch_size, seq_len, vocab_size]
        """
        return self.backend.get_hidden_states_and_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    @property
    def model(self):
        """Access underlying model"""
        return self.backend.model

    @property
    def tokenizer(self):
        """Access underlying tokenizer"""
        if not hasattr(self.backend, "tokenizer"):
            raise AttributeError(
                f"Backend '{self.backend_name}' does not have a tokenizer attribute"
            )
        if self.backend.tokenizer is None:
            raise ValueError(f"Backend '{self.backend_name}' does not have a tokenizer")
        return self.backend.tokenizer


def create_target_model(
    backend: str,
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    **extra_kwargs,
) -> TargetModelWrapper:
    """
    Factory function to create target model with appropriate backend configuration.

    Args:
        backend: Backend type, one of ["hf"]
        model_path: Path to model or serving endpoint URL
        torch_dtype: Data type for model weights (for HF backend)
        trust_remote_code: Whether to trust remote code
        tokenizer_path: Path to tokenizer
        **extra_kwargs: Additional backend-specific arguments

    Returns:
        TargetModelWrapper instance
    """
    # Prepare common kwargs
    kwargs = {"trust_remote_code": trust_remote_code, **extra_kwargs}

    # Add backend-specific kwargs
    if backend == "hf":
        kwargs.update(
            {
                "dtype": torch_dtype,
            }
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return TargetModelWrapper(backend=backend, model_path=model_path, **kwargs)
