import os
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from huggingface_hub import HfApi

from transformers import ResNetConfig, ResNetModel

from resnet10_forward import ResNet10

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader


# class ResNetConfig(PretrainedConfig):
#     def __init__(self, num_classes=1000, **kwargs):
#         super().__init__(**kwargs)
#         self.num_classes = num_classes

# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out

# class ResNet10(PreTrainedModel):
#     config_class = ResNetConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.in_channels = 64
        
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.layer1 = self._make_layer(64, 1)
#         self.layer2 = self._make_layer(128, 1, stride=2)
#         self.layer3 = self._make_layer(256, 1, stride=2)
#         self.layer4 = self._make_layer(512, 1, stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, config.num_classes)

#     def _make_layer(self, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(BasicBlock(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x
    

# class ResNet10Hf(ResNetModel):
#     config_class = ResNetConfig

#     def __init__(self, config):
#         super().__init__(config)
#         self.resnet = ResNet10(config)

#     def forward(self, x):
#         return self.resnet(x)

# class ResNetBlock(nn.Module):
#     """ResNet block."""

#     filters: int
#     conv: ModuleDef
#     norm: ModuleDef
#     act: Callable
#     strides: Tuple[int, int] = (1, 1)

#     @nn.compact
#     def __call__(
#         self,
#         x,
#     ):
#         residual = x
#         y = self.conv(self.filters, (3, 3), self.strides)(x)
#         y = self.norm()(y)
#         y = self.act(y)
#         y = self.conv(self.filters, (3, 3))(y)
#         y = self.norm()(y)

#         if residual.shape != y.shape:
#             residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
#                 residual
#             )
#             residual = self.norm(name="norm_proj")(residual)

#         return self.act(residual + y)
    
# # put inputs in [-1, 1]
# # x = observations.astype(jnp.float32) / 127.5 - 1.0
# if observations.shape[-3:-1] != self.image_size:
#     observations = resize(observations, self.image_size)

# # imagenet mean and std # TODO: add this back
# mean = jnp.array([0.485, 0.456, 0.406])
# std = jnp.array([0.229, 0.224, 0.225])
# x = (observations.astype(jnp.float32) / 255.0 - mean) / std

# conv = partial(
#     self.conv,
#     use_bias=False,
#     dtype=self.dtype,
#     kernel_init=nn.initializers.kaiming_normal(),
# )

# norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
# act = getattr(nn, 'relu')

# x = conv( -> embedder part
#     64,
#     (7, 7),
#     (2, 2),
#     padding=[(3, 3), (3, 3)],
#     name="conv_init",
# )(x)

# for i, block_size in enumerate((1, 1, 1, 1)):
#     for j in range(block_size):
#         print(i, j)
# x = norm(name="norm_init")(x)
# x = act(x)
# x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
# for i, block_size in enumerate((1, 1, 1, 1)):
#     for j in range(block_size):
#         x = self.block_cls(
#             self.num_filters * 2**i, # 64, 128, 256, 512
#             strides=stride, # (1, 1), (2, 2), (2, 2), (2, 2)
#             conv=conv,
#             norm=norm,
#             act=act,
#         )(x)
        
# return jax.lax.stop_gradient(x)
# # Extra
# height, width, channel = x.shape[-3:]
# x = SpatialLearnedEmbeddings(
#     height=height,
#     width=width,
#     channel=channel,
#     num_features=8,
# )(x)
# x = nn.Dropout(0.1, deterministic=not train)(x)
# x = nn.Dense(256)(x)
# x = nn.LayerNorm()(x)
# x = nn.tanh(x)

# class ResNetConvLayer(nn.Module):
#     def __init__(
#         self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
#     ):
#         super().__init__()
#         self.convolution = nn.Conv2d(
#             in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
#         )
#         self.normalization = nn.BatchNorm2d(out_channels)
#         self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

#     def forward(self, input: Tensor) -> Tensor:
#         hidden_state = self.convolution(input)
#         hidden_state = self.normalization(hidden_state)
#         hidden_state = self.activation(hidden_state)
#         return hidden_state
    
# class ResNetEmbeddings(nn.Module):
#     """
#     ResNet Embeddings (stem) composed of a single aggressive convolution.
#     """

#     def __init__(self, config: ResNetConfig):
#         super().__init__()
#         self.embedder = ResNetConvLayer(
#             config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act
#         )
#         self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.num_channels = config.num_channels

#     def forward(self, pixel_values: Tensor) -> Tensor:
#         num_channels = pixel_values.shape[1]
#         if num_channels != self.num_channels:
#             raise ValueError(
#                 "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
#             )
#         embedding = self.embedder(pixel_values)
#         embedding = self.pooler(embedding)
#         return embedding


# class ResNetModel(ResNetPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         self.embedder = ResNetEmbeddings(config)
#         self.encoder = ResNetEncoder(config)
#         self.pooler = nn.AdaptiveAvgPool2d((1, 1))
#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=BaseModelOutputWithPoolingAndNoAttention,
#         config_class=_CONFIG_FOR_DOC,
#         modality="vision",
#         expected_output=_EXPECTED_OUTPUT_SHAPE,
#     )
#     def forward(
#         self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
#     ) -> BaseModelOutputWithPoolingAndNoAttention:
#         embedding_output = self.embedder(pixel_values)

#         encoder_outputs = self.encoder(
#             embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
#         )

#         last_hidden_state = encoder_outputs[0]

#         pooled_output = self.pooler(last_hidden_state)

#         if not return_dict:
#             return (last_hidden_state, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPoolingAndNoAttention(
#             last_hidden_state=last_hidden_state,
#             pooler_output=pooled_output,
#             hidden_states=encoder_outputs.hidden_states,
#         )
    
# class ResNetEncoder(nn.Module):
#     def __init__(self, config: ResNetConfig):
#         super().__init__()
#         # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
#         self.stages = [
#             ResNetStage(
#                 config,
#                 64,
#                 64,
#                 stride=1,
#                 depth=1,
#             ),
#             ResNetStage(
#                 config,
#                 64,
#                 128,
#                 depth=1,
#             ),
#             ResNetStage(
#                 config,
#                 128,
#                 256,
#                 depth=1,
#             ),
#             ResNetStage(
#                 config,
#                 256,
#                 512,
#                 depth=1,
#             ),
#         ]

#         self.stages = nn.ModuleList(self.stages)

#     def forward(
#         self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
#     ) -> BaseModelOutputWithNoAttention:
#         for stage_module in self.stages:
#             hidden_state = stage_module(hidden_state)

#         if output_hidden_states:
#             hidden_states = hidden_states + (hidden_state,)

#         if not return_dict:
#             return tuple(v for v in [hidden_state, hidden_states] if v is not None)

#         return BaseModelOutputWithNoAttention(
#             last_hidden_state=hidden_state,
#             hidden_states=hidden_states,
#         )


BATCH_SIZE = 128
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def main():
    # Create and initialize the model
    #  Missed part of the code
    # 1.
    # put inputs in [-1, 1]
    # x = observations.astype(jnp.float32) / 127.5 - 1.0
    # if observations.shape[-3:-1] != self.image_size:
    #     observations = resize(observations, self.image_size)

    # # imagenet mean and std # TODO: add this back
    # mean = jnp.array([0.485, 0.456, 0.406])
    # std = jnp.array([0.229, 0.224, 0.225])
    # x = (observations.astype(jnp.float32) / 255.0 - mean) / std
    #
    # 2. Embedding has differnt normalization - BatchNorm2d instead of GroupNorm
    # 3. Different pooling method for Embedder - MaxPool2d sim vs Non sim

    config = ResNetConfig(
        num_channels=3,
        embedding_size=64,
        hidden_act="relu",
# -------
        hidden_sizes=[64, 128, 256, 512],  # Smaller hidden sizes for ResNet-10
        depths=[1, 1, 1, 1],  # One block per stage for ResNet-10
        layer_type="basic",    # ResNet-10 uses basic blocks
        downsample_in_first_stage=False
    )    

    model = ResNet10(config)
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

    model.eval()

    test_loss = 0.0
    test_labels = []
    test_pridections = []
    test_probs = []

    with torch.no_grad():
        for data in binary_testloader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(torch.float32).to(DEVICE)
            outputs = model(images)
            loss = model(outputs.logits, labels)
            test_loss += loss.item() * BATCH_SIZE

            test_labels.extend(labels.cpu())
            test_pridections.extend(outputs.logits.cpu())
            test_probs.extend(outputs.probabilities.cpu())

    test_loss = test_loss / len(binary_test_dataset)

    print(f"Test Loss: {test_loss:.4f}")
#     model.save_pretrained("resnet10")

# #  PreTrainedResNetEncoder(
# #                     pooling_method="spatial_learned_embeddings",
# #                     num_spatial_blocks=8,
# #                     bottleneck_dim=256,
# #                     pretrained_encoder=pretrained_encoder,
# #                     name=f"encoder_{image_key}",
# #                 )
#     #     "resnetv1-10-frozen": ft.partial(
#     #     ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock, pre_pooling=True, name="pretrained_encoder",
#     # ),
    
#     resnet10 = ResNetModel(config)
    
    # Save the model locally
    # model.save_pretrained("resnet10")
    
    # # Upload to Hugging Face Hub
    # # Replace 'your-username/resnet10' with your desired repository name
    # model.push_to_hub("helper2424/resnet10")
    
    print("Model uploaded successfully to Hugging Face Hub!")

if __name__ == "__main__":
    main()
