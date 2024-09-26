import torch
import torchvision
import torch.nn as nn
from torchinfo import summary


class MobilenetV2:
    def __init__(self, config):
        self.config = config

    def setup(self):
        pretrain = self.config["MODEL"]["PRE_TRAIN"]
        transfer_learning = self.config["MODEL"]["TRANSFER_LEARNING"]
        num_classes = int(self.config["MODEL"]["NUM_CLASSES"])
        retrain = self.config["SOLUTION"]["RETRAIN"]

        # get model from pytorch
        if pretrain == "True":
            pretrain_weights = (
                torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
            )
            self.torch_model = torchvision.models.mobilenet_v2(
                weights=pretrain_weights
            )
        else:
            self.torch_model = torchvision.models.mobilenet_v2()

        if retrain == "True" and num_classes != 1000:
            in_features = self.torch_model.classifier[-1].in_features
            self.torch_model.classifier[-1] = nn.Linear(
                in_features=in_features, out_features=num_classes
            )

        if retrain == "False" and num_classes != 1000:
            print(
                "[Warning] You ordered not to retrain but to change num_classes. Skipped changing num_classes."
            )

        # Transfer Learning  / Freeze parameters of backbone
        if transfer_learning == "True":
            for param in self.torch_model._parameters:
                param.requires_grad = False
            for param in self.torch_model.classifier._parameters:
                param.requires_grad = True

    def get(self):
        return self.torch_model

    def get_trained_model(self):
        return self.torch_model

    def show(self):
        input_size = int(self.config["MODEL"]["INPUT_SIZE"])
        summary(self.torch_model, input_size=(1, 3, input_size, input_size))

    def export_onnx(self, target_path):

        input_size = int(self.config["MODEL"]["INPUT_SIZE"])

        self.torch_model.to("cpu")

        x = torch.randn(1, 3, input_size, input_size, requires_grad=True)
        torch.onnx.export(
            self.torch_model,  # model being run
            x,  # model input (or a tuple for multiple inputs)
            target_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],  # the model's input names
            output_names=["output"],  # the model's output names
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
        )
