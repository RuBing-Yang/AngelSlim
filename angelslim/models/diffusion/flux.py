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

import torch
from diffusers import FluxPipeline

from ...utils.utils import find_layers
from ..base_model import BaseDiffusionModel
from ..model_factory import SlimModelFactory


@SlimModelFactory.register
class FLUX(BaseDiffusionModel):
    def __init__(
        self,
        model=None,
        deploy_backend="torch",
    ):
        super().__init__(
            model=model,
            deploy_backend=deploy_backend,
        )
        self.model_type = "flux"
        self.cache_helper = None

    def from_pretrained(
        self,
        model_path,
        torch_dtype="auto",
        cache_dir=None,
        use_cache_helper=False,
    ):
        """
        Load a pretrained FLUX model.
        Args:
            model_path (str): Path to the pretrained model.
            torch_dtype (str): Data type for the model weights.
            cache_dir (str): Directory to cache the model.
        """
        self.model = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        if use_cache_helper:
            self.model.cache_helper = self.cache_helper

    def generate(
        self,
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        seed=42,
    ):
        """
        Generate images using the FLUX model.
        Args:
            prompt (list): List of text prompt for image generation.
            height (int): Height of the generated images.
            width (int): Width of the generated images.
            guidance_scale (float): Guidance scale for the generation.
            num_inference_steps (int): Number of inference steps.
            max_sequence_length (int): Maximum sequence length for the model.
            seed (int): Random number torch.Generator for reproducibility.
        Returns:
            Generated image tensor.
        """
        generator = torch.Generator().manual_seed(seed)
        return self.model(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

    def get_observer_layers(self):
        names = [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "norm.linear",
            "proj_mlp",
            "proj_out",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_out",
            "to_out.0",
            "0.proj",
            "net.0",
            "net.2",
            "norm1.linear",
            "norm1_context.linear",
        ]
        self.quant_module = self.model.transformer
        observer_layers_dict = {}
        layers_dict = find_layers(self.quant_module, layers=self.observer_layer_classes)

        ignore_layers = self.skip_layer_names()
        for name, module in layers_dict.items():
            if self.block_name in name and (
                name.split(".")[-1] in names
                or name.split(".")[-2] + "." + name.split(".")[-1] in names
            ):
                observer_layers_dict[name] = module
            else:
                ignore_layers.append(name)
        self.quant_config.quant_algo_info["ignore_layers"] = ignore_layers

        return observer_layers_dict

    def get_quant_module(self):
        """
        Returns the module that will be quantized.
        This is typically the main transformer module of the model.
        """
        return self.model.transformer

    def get_quant_convert_module(self):
        """
        Returns the module that will be converted to quantized.
        This is typically the main transformer module of the model.
        """
        return self.model.transformer

    def get_save_func(self):
        pass
