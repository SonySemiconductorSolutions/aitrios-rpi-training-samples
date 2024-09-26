import numpy as np
import onnx
import onnxruntime
import torch
import mct_quantizers as mctq


class OnnxClassificationValidator:
    def __init__(self, config, quantize=True):
        self.config = config

        # path
        self.onnx_path = self.config["PATH"]["ONNX"]
        self.report_path = self.config["PATH"]["ONNX_SUMMARY"]
        self.quantized_onnx_path = self.config["PATH"]["QUANTIZED_ONNX"]
        self.quantized_report_path = self.config["PATH"][
            "QUANTIZED_ONNX_SUMMARY"
        ]

    def select_model(self, quantized=True):
        fpath = self.quantized_onnx_path if quantized else self.onnx_path
        self.model, self.session, self.input_name, self.output_name = (
            self.load_model(fpath, quantized)
        )

    def load_model(self, onnx_path, quantized=True):
        # load model
        print("loading onnx model:", onnx_path)
        model = onnx.load(onnx_path)
        PROVIDER = "CPUExecutionProvider"

        if quantized:
            session = onnxruntime.InferenceSession(
                onnx_path, mctq.get_ort_session_options(), providers=[PROVIDER]
            )
        else:
            session = onnxruntime.InferenceSession(
                model.SerializeToString(), providers=[PROVIDER]
            )

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        return model, session, input_name, output_name

    def predict_single_image(self, image):
        # if input is torch.tensor, convert to ndarray
        if torch.is_tensor(image):
            image = image.detach().numpy()
        # inference
        result = self.session.run(
            [self.output_name], {self.input_name: [image]}
        )

        return np.argmax(result)

    def validate_dataset(self, dataloader):
        correct_valid_sum = 0
        total_instances_valid = 0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.detach().numpy()
            results = self.session.run(
                [self.output_name], {self.input_name: inputs}
            )[0]
            # Accuracy
            results = torch.from_numpy(
                results.astype(np.float32)
            ).clone()  # to tensor
            classifications = torch.argmax(results, dim=1)
            correct_predictions = sum(classifications == labels).item()
            correct_valid_sum += correct_predictions
            total_instances_valid += len(inputs)

        acc_valid_average = correct_valid_sum / total_instances_valid

        return acc_valid_average

    def show_model(self, quantize=True):

        with open(self.report_path, "w") as f:
            onnx.checker.check_model(self.model)
            f.write(onnx.helper.printable_graph(self.model.graph))

    def validate(self, dataloader):

        self.select_model(quantized=False)
        before_acc = self.validate_dataset(dataloader)
        self.select_model(quantized=True)
        after_acc = self.validate_dataset(dataloader)

        print("Accuracy [before quantize]= ", before_acc)
        print("Accuracy [after quantize]= ", after_acc)

        return before_acc, after_acc
