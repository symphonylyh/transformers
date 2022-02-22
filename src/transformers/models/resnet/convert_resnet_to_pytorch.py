# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ResNet checkpoints from the original repository."""


import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import cached_download, hf_hub_url
from transformers import ConvNextFeatureExtractor, ResNetConfig, ResNetForImageClassification
from transformers.utils import logging
import timm

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

from functools import partial
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import List


@dataclass
class Tracker:
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = (
            len(list(m.modules())) == 1
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.BatchNorm2d)
        )
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))


@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def __call__(self, x: Tensor):
        """Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input.
        Under the hood we tracked all the operations in booth modules.
        :param x: [The input to the modules]
        :type x: torch.tensor
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")\

def print_named_parameters(m: nn.Module):
    for name, param in m.named_parameters():
        print(name, param.shape)
       

def convert_weights():
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000
    expected_shape = (1, num_labels)

    repo_id = "datasets/huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}


    ImageNetPreTrainedConfig = partial(ResNetConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_config = {
        "resnet18": ImageNetPreTrainedConfig(depths=[2,2,2,2]),
        "resnet26": ImageNetPreTrainedConfig(depths=[2,2,2,2], hidden_sizes=[64, 256, 512, 1024, 2048], layer_type="bottleneck"),
        "resnet34": ImageNetPreTrainedConfig(depths=[3, 4, 6, 3]),
        "resnet50": ImageNetPreTrainedConfig(depths=[3, 4, 6, 3], hidden_sizes=[64, 256, 512, 1024, 2048], layer_type="bottleneck"),
    }

    for name, config in names_to_config.items():
        from_model  = timm.create_model(name, pretrained=True)
        our_model = ResNetForImageClassification(config)
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)
        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)

        assert torch.allclose(from_model(x), our_model(x).logits)
   

    return config, expected_shape



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # Required parameters
    # parser.add_argument(
    #     "--checkpoint_url",
    #     default="https://dl.fbaipublicfiles.com/resnet/resnet_tiny_1k_224_ema.pth",
    #     type=str,
    #     help="URL of the original ConvNeXT checkpoint you'd like to convert.",
    # )
    # parser.add_argument(
    #     "--pytorch_dump_folder_path",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="Path to the output PyTorch model directory.",
    # )

    # args = parser.parse_args()
    convert_weights()
    # convert_resnet_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
