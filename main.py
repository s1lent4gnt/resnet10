#!/usr/bin/env python3
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoImageProcessor, AutoModel
from huggingface_hub import HfApi

from transformers import ResNetConfig, ResNetModel

from modeling_resnet import ResNet10

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
import argparse


BATCH_SIZE = 128
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def main():
    # Validate that model works as expected
    # Let's do a binary classification task
    # And check convergence
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The model name to download from the hub."
        ),
    )

    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_name)      

    model.to(DEVICE)

    target_binary_class = 3

    def one_vs_rest(dataset, target_class):
        new_targets = []
        for _, label in dataset:
            new_label = float(1.0) if label == target_class else float(0.0)
            new_targets.append(new_label)

        dataset.targets = new_targets  # Replace the original labels with the binary ones
        return dataset

    binary_train_dataset = CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    binary_test_dataset = CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    # Apply one-vs-rest labeling
    binary_train_dataset = one_vs_rest(binary_train_dataset, target_binary_class)
    binary_test_dataset = one_vs_rest(binary_test_dataset, target_binary_class)

    binary_trainloader = DataLoader(binary_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    binary_testloader = DataLoader(binary_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    binary_epoch = 1

    post_steps = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features=512, out_features=256),
        nn.LayerNorm(normalized_shape=256),
        nn.Tanh()
    )

    post_steps.to(DEVICE)

    model.eval()

    print(model)

    abort()
    test_loss = 0.0
    test_labels = []
    test_pridections = []
    test_probs = []

    with torch.no_grad():
        for data in binary_testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(torch.float32).to(DEVICE)
            outputs = model(images)
            outputs = post_steps(outputs)
            outputs = nn.Softmax(dim=1)(outputs)
            loss = model(outputs.logits, labels)
            test_loss += loss.item() * BATCH_SIZE

            test_labels.extend(labels.cpu())
            test_pridections.extend(outputs.logits.cpu())
            test_probs.extend(outputs.probabilities.cpu())

    test_loss = test_loss / len(binary_test_dataset)

    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
