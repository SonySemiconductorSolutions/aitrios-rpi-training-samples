import numpy as np
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import model_compression_toolkit as mct
import os
from datetime import datetime
import third_party.mct.coco_evaluation as coco_evaluation


class KerasObjectDetectionValidator:
    def __init__(self, config):
        self.config = config

        # path
        self.keras_path = self.config["PATH"]["KERAS"]
        self.report_path = self.config["PATH"]["KERAS_SUMMARY"]
        self.quantized_keras_path = self.config["PATH"]["QUANTIZED_KERAS"]
        self.quantized_report_path = self.config["PATH"][
            "QUANTIZED_KERAS_SUMMARY"
        ]
        self.valid_score_threshold = float(
            self.config["VALIDATOR"]["SCORE_THRESHOLD"]
        )
        self.data_path = self.config["PATH"]["DATA"]
        self.input_size = int(self.config["MODEL"]["INPUT_SIZE"])
        self.dataset_name = self.config["DATASET"]["NAME"]
        self.dataset_folder = self.config["PATH"]["DATASET"]
        self.annotation_file_path = self.config["PATH"]["ANNOTATION_FILE"]

        self.datetime = self.get_datetime()

    def select_model(self, quantized=True):
        if quantized:
            self.model = self.load_model(self.quantized_keras_path, quantized)
        else:
            self.model = self.load_model(self.keras_path, quantized)

    def load_model(self, keras_path, quantized=True):
        # load model
        print("loading keras model:", keras_path)

        if quantized:
            model = mct.keras_load_quantized_model(keras_path)
        else:
            model = load_model(keras_path)

        return model

    def predict_single_image(self, img_data, label_map):
        print("single_input_size:", img_data.shape)
        self.select_model(quantized=False)
        single_pred = self.model.predict(img_data)
        img = Image.open(self.single_image_path).convert("RGB")
        self.save_bbox_img(img, single_pred, label_map, quantized=False)

        self.select_model(quantized=True)
        self.save_bbox_img(img, single_pred, label_map, quantized=True)

        return None

    def save_bbox_img(self, img, pred, label_map, quantized=True):
        image_size = img.size
        valid_detections = pred[1] > self.valid_score_threshold
        draw_img = ImageDraw.Draw(img)
        if quantized:
            rectcolor = (255, 255, 0)
        else:
            rectcolor = (0, 255, 0)

        textcolor = (0, 0, 0)

        for i, is_detection in enumerate(valid_detections[0]):
            if is_detection:
                normalized_bbox = np.array(pred[0][0][i])
                bbox = self.yxyx2xywh(normalized_bbox, image_size)
                label = label_map[int(pred[2][0][i])]
                self.show_bbox(draw_img, bbox, label, textcolor, rectcolor)
            else:
                break

        if not (os.path.isdir("results")):
            os.mkdir("results")
        self.results_path = "results/" + self.datetime
        if not (os.path.isdir(self.results_path)):
            os.mkdir(self.results_path)

        if quantized:
            img.save(
                self.results_path + "/quantized_output.jpg",
                "JPEG",
                quality=95,
            )
        else:
            img.save(
                self.results_path + "/output.jpg",
                "JPEG",
                quality=95,
            )

    def show_bbox(self, draw, bbox, text, textcolor, bbcolor):
        font = ImageFont.truetype("arial.ttf", 10)
        textbox = draw.multiline_textbbox((bbox[0], bbox[1]), text, font)
        bbox_w = bbox[2]
        bbox_h = bbox[3]
        text_w = textbox[2] - textbox[0]
        text_h = textbox[3] - textbox[1]
        textarea = (bbox[0], bbox[1], textbox[2], textbox[3])
        draw.rectangle(
            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
            outline=bbcolor,
        )
        draw.rectangle(
            textarea,
            outline=bbcolor,
            fill=bbcolor,
        )
        draw.text((bbox[0], bbox[1]), text, fill=textcolor, font=font)

    def validate_dataset(self, dataloader, quantized):
        coco_metric = coco_evaluation.CocoEval(self.annotation_file_path)
        if self.dataset_name == "CocoVal2017":
            is_coco = True
        else:
            is_coco = False

        for images, targets in dataloader:
            outputs = self.model(images)
            coco_metric.add_batch_detections(
                outputs, 
                targets, 
                is_coco, 
                self.valid_score_threshold
            )

        return coco_metric.result(quantized)[0]

    def show_model(self, quantized=True):
        if quantized:
            with open(self.quantized_report_path, "w") as f:
                self.model.summary(print_fn=lambda x: f.write(x + "\r\n"))
        else:
            with open(self.report_path, "w") as f:
                self.model.summary(print_fn=lambda x: f.write(x + "\r\n"))

    def yxyx2xywh(self, normalized_bbox, image_size):
        normalized_bbox[0], normalized_bbox[1] = (
            normalized_bbox[1],
            normalized_bbox[0],
        )
        normalized_bbox[2], normalized_bbox[3] = (
            normalized_bbox[3],
            normalized_bbox[2],
        )

        x = normalized_bbox[0] * image_size[0]
        y = normalized_bbox[1] * image_size[1]

        x2 = normalized_bbox[2] * image_size[0]
        y2 = normalized_bbox[3] * image_size[1]

        w = round(x2 - x, 2)
        h = round(y2 - y, 2)
        x = round(x, 2)
        y = round(y, 2)

        bbox = (x, y, w, h)
        return bbox

    def validate(self, dataloader):
        self.select_model(quantized=False)
        before_mAP = self.validate_dataset(dataloader, quantized=False)

        self.select_model(quantized=True)
        after_mAP = self.validate_dataset(dataloader, quantized=True)

        return before_mAP, after_mAP

    def get_datetime(self):
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return dt
