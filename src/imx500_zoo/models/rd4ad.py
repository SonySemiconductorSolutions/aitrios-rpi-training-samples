import torch
import torch.nn as nn
from torchvision import models
from torch import Tensor
from typing import Optional, Callable


class MobileNetV2Encoder(nn.Module):
    def __init__(self, weights=None):
        super(MobileNetV2Encoder, self).__init__()
        mobilenet_v2 = models.mobilenet_v2(weights=weights)
        self.feature_a_layers = mobilenet_v2.features[:7]
        self.feature_b_layers = mobilenet_v2.features[7:14]
        self.feature_c_layers = mobilenet_v2.features[14:-1]

    def forward(self, x):
        feature_a = self.feature_a_layers(x)
        feature_b = self.feature_b_layers(feature_a)
        feature_c = self.feature_c_layers(feature_b)

        return [feature_a, feature_b, feature_c]


class BNLayer(nn.Module):
    def __init__(
            self,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            use_mff=False):
        super(BNLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = norm_layer(64)

        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.bn3 = norm_layer(96)

        self.conv4 = nn.Conv2d(480, 320, kernel_size=1, stride=1)
        self.bn4 = norm_layer(320)

        self.use_mff = use_mff
        if self.use_mff:
            self.scale_weights = nn.Parameter(torch.ones(3))

    def forward(self, x: Tensor) -> Tensor:
        l1 = self.relu(self.bn1(self.conv1(x[0])))
        l2 = self.relu(self.bn2(self.conv2(l1)))
        l3 = self.relu(self.bn3(self.conv3(x[1])))

        if self.use_mff:
            l2_scaled = self.scale_weights[0] * l2
            l3_scaled = self.scale_weights[1] * l3
            x2_scaled = self.scale_weights[2] * x[2]
            feature = torch.cat([l2_scaled, l3_scaled, x2_scaled], dim=1)
        else:
            feature = torch.cat([l2, l3, x[2]], dim=1)

        output = self.bn4(self.conv4(feature))
        return output.contiguous()


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and inp == oup)
        hidden_dim = int(round(inp * expand_ratio))

        self.conv = self._make_layers(inp, hidden_dim, oup, stride)

    def _make_layers(self, inp, hidden_dim, oup, stride):
        layers = [
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ]

        if stride > 1:
            layers.append(nn.Upsample(scale_factor=stride,
                          mode='bilinear', align_corners=False))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res_connect else out


class DeMobileNetV2Decoder(nn.Module):
    def __init__(self, inverted_residual_setting, last_channel=320):
        super(DeMobileNetV2Decoder, self).__init__()

        input_channel = last_channel
        features = []

        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            expand_ratio = t
            for i in range(n):
                stride_adjusted = s if i == 0 else 1
                features.append(InvertedResidual(
                    input_channel,
                    output_channel,
                    stride_adjusted,
                    expand_ratio
                ))
                input_channel = output_channel

        self.features = nn.Sequential(*features)

    def forward(self, x: Tensor) -> Tensor:
        feature_c = None
        feature_b = None
        feature_a = None

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 0:
                feature_c = x
            elif idx == 6:
                feature_b = x
            elif idx == 13:
                feature_a = x
                dummy = x + 1

        return [feature_a, feature_b, feature_c, dummy]


def mobilenet_v2_encoder(
        weights='openimagesv7',
        freeze_encoder=True,
        use_mff=False,
        weight_filename=None):
    encoder = MobileNetV2Encoder(weights=None)

    if weights == 'openimagesv7':
        weight_path = (
            weight_filename or
            './pretrained_weights/rd4ad/rd4ad_encoder_final.pth'
        )
        print(f'Load pre-trained encoder from: {weight_path}')
        checkpoint = torch.load(weight_path)
        encoder.load_state_dict(checkpoint)
    else:
        raise ValueError("Unsupported weight type")

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    bn = BNLayer(use_mff=use_mff)
    return encoder, bn


class RD4ADModel(nn.Module):
    def __init__(self, config):
        super(RD4ADModel, self).__init__()
        self.config = config

        is_retrain = config["SOLUTION"].getboolean("RETRAIN", True)

        if not is_retrain:
            target_object = config["DATASET"].get("OBJECT", "pipe_fryum")
            pretrained_weights_path = (
                f'./pretrained_weights/rd4ad/rd4ad_{target_object}_final.pth'
            )
            print(
                f"Load pre-trained RD4AD model for {target_object} "
                f"from: {pretrained_weights_path}"
            )

            checkpoint = torch.load(
                pretrained_weights_path, map_location=torch.device("cpu"))
            self.encoder = MobileNetV2Encoder(weights=None)
            self.bn = BNLayer()
            self.decoder = DeMobileNetV2Decoder(self._get_decoder_setting())

            self.encoder.load_state_dict({
                k.replace("encoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("encoder.")
            })
            self.bn.load_state_dict({
                k.replace("bn.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("bn.")
            })
            self.decoder.load_state_dict({
                k.replace("decoder.", ""): v
                for k, v in checkpoint.items()
                if k.startswith("decoder.")
            })
        else:
            use_mff = config["MODEL"].getboolean("USE_WMFF", False)

            self.encoder, self.bn = mobilenet_v2_encoder(
                weights="openimagesv7",
                freeze_encoder=True,
                use_mff=use_mff,
                weight_filename=None
            )

            self.decoder = DeMobileNetV2Decoder(self._get_decoder_setting())

    def _get_decoder_setting(self):
        return [
            [6, 320, 1, 1], [6, 160, 3, 1], [6, 96, 3, 2],
            [6, 64, 4, 1], [6, 32, 3, 2], [6, 24, 2, 2], [1, 16, 1, 2]
        ]

    def forward(self, x):
        features = self.encoder(x)
        normalized_features = self.bn(features)
        reconstructed = self.decoder(normalized_features)
        return features, reconstructed

    def setup(self):
        pass

    def get_trained_model(self):
        return self

    def export_onnx(self, target_path):
        input_size = int(self.config["MODEL"].get("INPUT_SIZE", 256))

        self.to("cpu")

        x = torch.randn(1, 3, input_size, input_size, requires_grad=True)
        torch.onnx.export(
            self,
            x,
            target_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=[
                'feature_a', 'feature_b', 'feature_c',
                'reconstructed_a', 'reconstructed_b',
                'reconstructed_c', 'dummy'
            ],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'feature_a': {0: 'batch_size'},
                'feature_b': {0: 'batch_size'},
                'feature_c': {0: 'batch_size'},
                'reconstructed_a': {0: 'batch_size'},
                'reconstructed_b': {0: 'batch_size'},
                'reconstructed_c': {0: 'batch_size'},
                'dummy': {0: 'batch_size'}
            },
        )

        if torch.cuda.is_available():
            self.to("cuda")
