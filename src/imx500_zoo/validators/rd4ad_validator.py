import numpy as np
import onnx
import onnxruntime
import torch
import mct_quantizers as mctq
import os
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from imx500_zoo.models.rd4ad import RD4ADModel
from third_party.rd4ad.test import (
    cal_anomaly_map,
    compute_pro,
    min_max_norm,
    cvt2heatmap,
    show_cam_on_image
)
import cv2
from tqdm import tqdm


class RD4ADValidator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.onnx_path = config["PATH"]["ONNX"]
        self.quantized_onnx_path = config["PATH"]["QUANTIZED_ONNX"]
        self.pytorch_path = config["PATH"]["MODEL"] + \
            config["SOLUTION"]["NAME"] + ".pth"
        self.do_pytorch = config["VALIDATOR"].getboolean("DO_PYTORCH", False)

        self.model = None
        self.session = None
        self.input_name = None
        self.output_names = None

    def load_pytorch_model(self):
        model = RD4ADModel(self.config)

        is_retrain = self.config["SOLUTION"].getboolean("RETRAIN", True)
        if is_retrain:
            weight_path = self.pytorch_path
        else:
            target_object = self.config["DATASET"].get("OBJECT", "pipe_fryum")
            weight_path = (
                f'./pretrained_weights/rd4ad/rd4ad_{target_object}_final.pth'
            )

        print(f"ReLoad Pytorch model from: {weight_path}")
        model.load_state_dict(torch.load(
            weight_path, map_location=self.device))
        self.model = model.to(self.device)
        self.model.eval()

    def select_model(self, quantized=True):
        fpath = self.quantized_onnx_path if quantized else self.onnx_path
        self.model, self.session, self.input_name, self.output_names = (
            self.load_model(fpath, quantized)
        )

    def load_model(self, onnx_path, quantized=True):
        model_type = "Quantized ONNX model" if quantized else "ONNX model"
        print(f"Load {model_type} from: {onnx_path}")

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
        output_names = [output.name for output in session.get_outputs()]

        return model, session, input_name, output_names

    def predict_pytorch(self, images):
        images = images.to(self.device)
        with torch.no_grad():
            features, reconstructed = self.model(images)
        return features, reconstructed

    def predict_onnx(self, images):
        images = images.cpu().numpy() if torch.is_tensor(images) else images
        output = self.session.run(self.output_names, {self.input_name: images})

        features = [torch.tensor(output[i]) for i in range(3)]
        reconstructed = [torch.tensor(output[i]) for i in range(3, 7)]
        return features, reconstructed

    def validate_model(self, dataloader, model_type="Pytorch"):
        model_predict = (
            self.predict_pytorch
            if model_type == "Pytorch"
            else self.predict_onnx
        )

        gt_list_px, pr_list_px = [], []
        gt_list_sp, pr_list_sp = [], []
        aupro_list = []
        total_batches = len(dataloader)
        print(f"Starting {model_type} model validation...")

        with torch.no_grad():
            for img, gt, label, _ in tqdm(
                dataloader, total=total_batches, desc="Validating"
            ):
                features, reconstructed = model_predict(img)

                anomaly_map, _ = cal_anomaly_map(
                    features, reconstructed, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0

                if label.item() != 0 and np.any(gt.cpu().numpy() == 1):
                    aupro_list.append(
                        compute_pro(
                            gt.squeeze(0).cpu().numpy().astype(int),
                            anomaly_map[np.newaxis, :, :]
                        ))

                gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                pr_list_px.extend(anomaly_map.ravel())
                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))

            auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
            auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
            aupro_px = round(np.mean(aupro_list), 3)

        print(
            f"[{model_type} Model] "
            f"Seg-AUROC: {auroc_px * 100:.2f}%, "
            f"Det-AUROC: {auroc_sp * 100:.2f}%, "
            f"AUPRO: {aupro_px * 100:.2f}%\n")

        return auroc_px, auroc_sp, aupro_px

    def visualization(self, model_path, dataloader, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        session_options = mctq.get_ort_session_options()
        session = onnxruntime.InferenceSession(
            model_path, session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        count = 0
        print("Visualization started...")
        total_anomalies = sum(
            1 for _, _, label, _ in dataloader if label.item() == 1)
        with torch.no_grad():
            for img, gt, label, _ in tqdm(
                dataloader, total=total_anomalies,
                desc="Processing anomalies"
            ):
                if label.item() == 0:
                    continue

                img_np = img.cpu().numpy()
                outputs = session.run(output_names, {input_name: img_np})

                features = [torch.tensor(output) for output in outputs[:3]]
                reconstructed = [torch.tensor(output)
                                 for output in outputs[3:]]

                anomaly_map, _ = cal_anomaly_map(
                    features, reconstructed, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)

                anomaly_map = min_max_norm(anomaly_map)
                heatmap = cvt2heatmap(anomaly_map * 255)

                img_visual = cv2.cvtColor(
                    img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255,
                    cv2.COLOR_BGR2RGB
                )
                img_visual = np.uint8(min_max_norm(img_visual) * 255)

                overlayed_image = show_cam_on_image(img_visual, heatmap)

                cv2.imwrite(os.path.join(
                    output_dir, f"{count}_org.png"), img_visual)
                cv2.imwrite(os.path.join(
                    output_dir, f"{count}_ad.png"), overlayed_image)

                gt_image = gt[0].cpu().numpy().astype(np.uint8) * 255
                if gt_image.ndim > 2:
                    gt_image = gt_image.squeeze()

                cv2.imwrite(os.path.join(
                    output_dir, f"{count}_gt.png"), gt_image)

                count += 1

    def validate(self, dataloader):
        if self.do_pytorch:
            self.load_pytorch_model()
            _ = self.validate_model(dataloader, model_type="Pytorch")

        self.select_model(quantized=False)
        onnx_results = self.validate_model(dataloader, model_type="ONNX")

        self.select_model(quantized=True)
        quantized_onnx_results = self.validate_model(
            dataloader, model_type="Quantized ONNX")

        if self.config["VALIDATOR"].getboolean("VISUALIZE", False):
            visualization_dir = os.path.join(
                self.config["PATH"]["MODEL"], "visualization")
            self.visualization(self.quantized_onnx_path,
                               dataloader, visualization_dir)
            print(f"Visualization images saved to: {visualization_dir}")

        return onnx_results, quantized_onnx_results
