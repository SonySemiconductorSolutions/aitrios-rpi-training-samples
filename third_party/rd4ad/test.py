import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(fs_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        if fs.shape[-2:] != ft.shape[-2:]:
            target_size = fs.shape[-2:]
            ft = F.interpolate(ft, size=target_size,
                               mode='bilinear', align_corners=True)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size,
                              mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def evaluation(integrated_model, dataloader, device, _class_=None):
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:

            img = img.to(device)
            integrated_model.eval()
            inputs, outputs = integrated_model(img)
            anomaly_map, _ = cal_anomaly_map(
                inputs, outputs, img.shape[-1], amap_mode='a')
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
    return auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """
    Compute the area under the curve of per-region overlaping (PRO)
    and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test.
        masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test.
        amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    masks = (masks > 0.5).astype(int)

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, (
        "amaps.shape and masks.shape must be same"
    )
    if set(masks.flatten()) != {0, 1}:
        print("Warning: masks do not contain both 0 and 1")
        return 0.0
    assert set(masks.flatten()) == {0, 1}, (
        "set(masks.flatten()) must be {0, 1}"
    )
    assert isinstance(num_th, int), "type(num_th) must be int"

    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    if min_th == max_th:
        thresholds = [min_th - 0.01, min_th + 0.01]
    else:
        delta = (max_th - min_th) / num_th
        thresholds = np.arange(min_th, max_th, delta)

    data_list = []
    for th in thresholds:
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        if len(pros) > 0:
            data_list.append({"pro": mean(pros), "fpr": fpr, "threshold": th})

    df = pd.DataFrame(data_list)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()
    if len(df) < 2:
        print("Not enough points to compute AUC")
        return 0.0

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def detection(encoder, bn, decoder, dataloader, device, _class_):
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(
                inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
