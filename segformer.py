from typing import Any, Optional, Union, Callable

from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.base.model import init

from segmentation_models_pytorch.base.heads import Activation
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.hub_mixin import supports_config_loading

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

# model building code taken from smp module to avoid editing whole module

class MLP(nn.Module):
    def __init__(self, skip_channels, segmentation_channels):
        super().__init__()

        self.linear = nn.Linear(skip_channels, segmentation_channels)

    def forward(self, x: torch.Tensor):
        batch, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        return x


class SegformerDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        segmentation_channels=256,
    ):
        super().__init__()

        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for Segformer decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        if encoder_channels[1] == 0:
            encoder_channels = tuple(
                channel for index, channel in enumerate(encoder_channels) if index != 1
            )
        encoder_channels = encoder_channels[::-1]

        self.mlp_stage = nn.ModuleList(
            [MLP(channel, segmentation_channels) for channel in encoder_channels[:-1]]
        )

        self.fuse_stage = md.Conv2dReLU(
            in_channels=(len(encoder_channels) - 1) * segmentation_channels,
            out_channels=segmentation_channels,
            kernel_size=1,
            use_batchnorm=True,
        )

    def forward(self, *features):
        # Resize all features to the size of the largest feature
        target_size = [dim // 4 for dim in features[0].shape[2:]]

        features = features[2:] if features[1].size(1) == 0 else features[1:]
        features = features[::-1]  # reverse channels to start from head of encoder

        resized_features = []
        for feature, stage in zip(features, self.mlp_stage):
            feature = stage(feature)
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_features.append(resized_feature)

        output = self.fuse_stage(torch.cat(resized_features, dim=1))

        return output
    
    
class Segformer(SegmentationModel):
    """Segformer is simple and efficient design for semantic segmentation with Transformers

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks, default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        kwargs: Arguments passed to the encoder class ``__init__()`` function. Applies only to ``timm`` models. Keys with ``None`` values are pruned before passing.

    Returns:
        ``torch.nn.Module``: **Segformer**

    .. _Segformer:
        https://arxiv.org/abs/2105.15203

    """

    @supports_config_loading
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_segmentation_channels: int = 256,
        in_channels: int = 3,
        activation: Optional[Union[str, Callable]] = None,
        aux_params: Optional[dict] = None,
        **kwargs: dict[str, Any],
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **kwargs,
        )

        self.decoder = SegformerDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            segmentation_channels=decoder_segmentation_channels,
        )

        # we don't need aux head here
        self.classification_head = None

        # this is new - define multiple heads and overwrite forward pass
        self.seg_binary_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=2,
            activation=activation,
            kernel_size=1,
            upsampling=4,
        )
        
        self.seg_triple_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=3,
            activation=activation,
            kernel_size=1,
            upsampling=4,
        )      
          
        self.seg_full_head = SegmentationHead(
            in_channels=decoder_segmentation_channels,
            out_channels=7,
            activation=activation,
            kernel_size=1,
            upsampling=4,
        )
        
        self.name = "segformer-{}".format(encoder_name)
        self.initialize()

        
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_binary_head)
        init.initialize_head(self.seg_triple_head)
        init.initialize_head(self.seg_full_head)

    
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not torch.jit.is_tracing() or self.requires_divisible_input_shape:
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        # here we just forward decoder output into multiple heads and return them
        # one by one
        
        binary_mask = self.seg_binary_head(decoder_output)
        triple_mask = self.seg_triple_head(decoder_output)
        full_mask = self.seg_full_head(decoder_output)


        return binary_mask, triple_mask, full_mask