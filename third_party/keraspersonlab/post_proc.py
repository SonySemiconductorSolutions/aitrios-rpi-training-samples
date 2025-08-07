import numpy as np

from scipy.sparse import coo_matrix
from scipy.ndimage import maximum_filter, gaussian_filter

input_shape = None
config = None


def set_config(config_i):
    global config
    global input_shape

    config = config_i
    input_shape = [config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]]


def iterative_bfs(graph, start, path=[]):
    """iterative breadth first search from start"""
    q = [(None, start)]
    visited = []
    while q:
        v = q.pop(0)
        if not v[1] in visited:
            visited.append(v[1])
            path = path + [v]
            q = q + [(v[1], w) for w in graph[v[1]]]
    return path


def accumulate_votes(votes, shape):
    xs = votes[:, 0]
    ys = votes[:, 1]
    ps = votes[:, 2]
    tl = [np.floor(ys).astype("int32"), np.floor(xs).astype("int32")]
    tr = [np.floor(ys).astype("int32"), np.ceil(xs).astype("int32")]
    bl = [np.ceil(ys).astype("int32"), np.floor(xs).astype("int32")]
    br = [np.ceil(ys).astype("int32"), np.ceil(xs).astype("int32")]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps * (1.0 - dx) * (1.0 - dy)
    tr_vals = ps * dx * (1.0 - dy)
    bl_vals = ps * dy * (1.0 - dx)
    br_vals = ps * dy * dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    heatmap = np.asarray(
        coo_matrix(
            (data[good_inds], (I[good_inds], J[good_inds])), shape=shape
        ).todense()
    )
    return heatmap


def compute_heatmaps(kp_maps, short_offsets):
    heatmaps = []
    map_shape = kp_maps.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1, 0, 2))
    for i in range(config.NUM_KP):
        this_kp_map = kp_maps[:, :, i : i + 1]
        votes = idx + short_offsets[:, :, 2 * i : 2 * i + 2]
        votes = np.reshape(
            np.concatenate([votes, this_kp_map], axis=-1), (-1, 3)
        )
        heatmaps.append(
            accumulate_votes(votes, shape=map_shape)
            / (np.pi * config.KP_RADIUS**2)
        )

    return np.stack(heatmaps, axis=-1)


def get_keypoints(heatmaps):
    keypoints = []
    for i in range(config.NUM_KP):
        peaks = (
            maximum_filter(
                heatmaps[:, :, i], footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            )
            == heatmaps[:, :, i]
        )
        peaks = zip(*np.nonzero(peaks))
        keypoints.extend(
            [
                {
                    "id": i,
                    "xy": np.array(peak[::-1]),
                    "conf": heatmaps[peak[0], peak[1], i],
                }
                for peak in peaks
            ]
        )
        keypoints = [kp for kp in keypoints if kp["conf"] > config.PEAK_THRESH]

    return keypoints


def get_keypoints_metrics(heatmaps_target, heatmaps_output):
    keypoints_target = {}
    keypoints_output = {}

    for i in range(config.NUM_KP):
        peaks_target = (
            maximum_filter(
                heatmaps_target[:, :, i],
                footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            )
            == heatmaps_target[:, :, i]
        )
        peaks_target = zip(*np.nonzero(peaks_target))

        peaks_output = heatmaps_output[:, :, i].copy()

        keypoints_target[str(i)] = []
        keypoints_output[str(i)] = []

        for peak in peaks_target:
            idx = i
            conf = heatmaps_target[peak[0], peak[1], i]
            if conf > config.PEAK_THRESH:
                xy = list(peak[::-1])
                xy.append(conf)
                keypoints_target[str(i)].append(xy)

        cnt = 0

        while cnt < len(keypoints_target[str(i)]):

            if peaks_output.max() > config.PEAK_THRESH:

                y, x = np.where(peaks_output == peaks_output.max())
                if len(x) > 0 and len(y) > 0:
                    confo = peaks_output[y, x][0]
                    xyo = [int(x[0]), int(y[0])]
                    xyo.append(confo)
                    keypoints_output[str(i)].append(xyo)

                else:
                    xyo = [0, 0]
                    confo = 0
                    xyo.append(confo)
                    keypoints_output[str(i)].append(xyo)

                peaks_output[y, x] = -9999

            else:
                xyo = [0, 0]
                confo = 0
                xyo.append(confo)
                keypoints_output[str(i)].append(xyo)

            cnt += 1

    return keypoints_target, keypoints_output


def computeOks(target_keypoint_dict, gts, dts, summary_dict, keypoint_class):

    kpt_oks_sigmas = config.kpt_oks_sigmas
    maxDets = config.maxDets

    gts = np.array(gts)
    dts = np.array(dts)

    visibility = gts[:, 2]
    total_valid_gts = visibility[visibility > 0]
    total_gt_count = total_valid_gts.shape[0]
    total_valid_dt_count = 0
    FP = 0
    TP = 0
    e = 0  # 0 initial value for not passing oks thresh if not visible
    if len(gts) == 0 or len(dts) == 0:
        # if need to calculate FN, can do it here using dts
        summary_dict[keypoint_class].extend([0, 0])
        return summary_dict

    inds = np.argsort([-d for d in dts[:, 2]], kind="mergesort")
    dts = [dts[i] for i in inds]

    if len(dts) > maxDets[-1]:
        dts = dts[0 : maxDets[-1]]

    # ious = np.zeros((len(dts), len(gts)))
    sigmas = kpt_oks_sigmas
    vars_kp = (sigmas * 2) ** 2
    vars = vars_kp[int(keypoint_class)]
    k = len(sigmas)

    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):

        g = gt
        xg = g[0::3]
        yg = g[1::3]
        vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)

        for i, dt in enumerate(dts):
            d = dt
            xd = d[0::3]
            yd = d[1::3]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
                # added
                e = (
                    (dx**2 + dy**2)
                    / vars
                    / (
                        np.sqrt(target_keypoint_dict["area"][j])
                        + np.spacing(1)
                    )
                    / 2
                )

            # uncomment to store oks values in array
            # ious[j, i] = np.sum(np.exp(-e)) / e.shape[0]
            # ious[ious<config.OKS_THRESH]=0
            e = np.exp(-e)
            if e < config.OKS_THRESH:
                total_valid_dt_count += 1

            # add end

    if total_valid_dt_count > total_gt_count:
        FP = total_valid_dt_count - total_gt_count
        TP = total_gt_count
    else:
        TP = total_valid_dt_count
        FP = 0
        FN = total_gt_count - total_valid_dt_count

    summary_dict[keypoint_class].extend([TP, FP])

    return summary_dict


def get_keypoints_oks_metrics(target_keypoint_dict, heatmaps_output):
    keypoints_target = {}
    keypoints_output = {}
    target_keypoint_list = target_keypoint_dict["keypoints"]
    summary_dict = {}

    for i in range(config.NUM_KP):

        peaks_output = heatmaps_output[:, :, i].copy()
        # suppress all pred which is below conf threshold
        peaks_output[peaks_output < config.PEAK_THRESH] = 0
        peaks_output = zip(*np.nonzero(peaks_output))

        keypoints_target[str(i)] = []
        keypoints_output[str(i)] = []
        summary_dict[str(i)] = []

        for item in target_keypoint_list:
            keypoints_target[str(i)].append(item[i])

        for peak in peaks_output:
            idx = i
            conf = heatmaps_output[peak[0], peak[1], i]
            xy = list(peak[::-1])
            xy.append(conf)
            keypoints_output[str(i)].append(xy)

        summary_dict = computeOks(
            target_keypoint_dict,
            keypoints_target[str(i)],
            keypoints_output[str(i)],
            summary_dict,
            str(i),
        )

    return summary_dict


def computePCK(
    target_keypoint_dict,
    gts,
    dts,
    summary_dict,
    keypoint_class,
    torso_dia_list,
):

    kpt_oks_sigmas = config.kpt_oks_sigmas
    maxDets = config.maxDets

    gts = np.array(gts)
    dts = np.array(dts)

    visibility = gts[:, 2]
    total_valid_gts = visibility[visibility > 0]
    total_gt_count = total_valid_gts.shape[0]
    total_valid_dt_count = 0
    FP = 0
    TP = 0
    e = 9999  # high initial value for not passing pck thresh
    if len(gts) == 0 or len(dts) == 0:
        # if need to calculate FN, can do it here using dts
        summary_dict[keypoint_class].extend([0, 0])
        return summary_dict

    inds = np.argsort([-d for d in dts[:, 2]], kind="mergesort")
    dts = [dts[i] for i in inds]

    if len(dts) > maxDets[-1]:
        dts = dts[0 : maxDets[-1]]

    # ious = np.zeros((len(dts), len(gts)))

    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):

        g = gt
        xg = g[0::3]
        yg = g[1::3]
        vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)

        for i, dt in enumerate(dts):
            d = dt
            xd = d[0::3]
            yd = d[1::3]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
                # added
                e = (dx**2 + dy**2) ** 0.5

            # uncomment to store L2 values in array
            # ious[j, i] = np.sum(np.exp(-e)) / e.shape[0]
            # ious[ious<config.OKS_THRESH]=0
            if e < torso_dia_list[j]:
                total_valid_dt_count += 1

            # add end

    if total_valid_dt_count > total_gt_count:
        FP = total_valid_dt_count - total_gt_count
        TP = total_gt_count
    else:
        TP = total_valid_dt_count
        FP = 0
        FN = total_gt_count - total_valid_dt_count

    summary_dict[keypoint_class].extend([TP, FP])

    return summary_dict


def get_keypoints_pck_metrics(target_keypoint_dict, heatmaps_output):
    keypoints_target = {}
    keypoints_output = {}
    target_keypoint_list = target_keypoint_dict["keypoints"]
    torso_dia_list = []

    summary_dict = {}

    for i in range(config.NUM_KP):

        peaks_output = heatmaps_output[:, :, i].copy()
        # suppress all pred which is below conf threshold
        peaks_output[peaks_output < config.PEAK_THRESH] = 0
        peaks_output = zip(*np.nonzero(peaks_output))

        keypoints_target[str(i)] = []
        keypoints_output[str(i)] = []
        summary_dict[str(i)] = []

        for item in target_keypoint_list:
            keypoints_target[str(i)].append(item[i])
            # torso diameter- distance between left shoulder and right hip of each ground-truth pose
            torso_dia = config.get_torso_dia(item, input_shape)
            torso_dia_list.append(torso_dia)

        for peak in peaks_output:
            idx = i
            conf = heatmaps_output[peak[0], peak[1], i]
            xy = list(peak[::-1])
            xy.append(conf)
            keypoints_output[str(i)].append(xy)

        summary_dict = computePCK(
            target_keypoint_dict,
            keypoints_target[str(i)],
            keypoints_output[str(i)],
            summary_dict,
            str(i),
            torso_dia_list,
        )

    return summary_dict


def computeOks_full(target_keypoint_dict, gts, dts):

    if len(gts) == 0:
        # no gt values for image, need to omit in metrics
        return [-1, -1, -1]

    kpt_oks_sigmas = config.kpt_oks_sigmas
    total_valid_gt_count = 0
    total_valid_dt_count = 0
    total_fp_dt_count = 0
    total_tp_dt_count = 0
    FN = 0

    oks = np.zeros((len(dts), len(gts)))
    sigmas = kpt_oks_sigmas
    vars = (sigmas * 2) ** 2
    k = len(sigmas)

    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):

        e = np.array([0])
        g = np.array(gt)
        xg = g[:, 0]
        yg = g[:, 1]
        vg = g[:, 2]
        k1 = np.count_nonzero(vg > 0)
        total_valid_gt_count += k1
        for i, dt in enumerate(dts):
            d = np.array(dt)
            xd = d[:, 0]
            yd = d[:, 1]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
                # added
                e = (
                    (dx**2 + dy**2)
                    / vars
                    / ((target_keypoint_dict["area"][j]) + np.spacing(1))
                    / 2
                )
                # e=np.exp(-e)
                e = e[vg > 0]
            # return mean of below oks if needed instead of oks-AR/AP
            oks = np.sum(np.exp(-e)) / e.shape[0]
            # since good detection means oks close to 1
            # total_valid_dt_count+=e[e>config.OKS_THRESH].shape[0]
            if oks > config.OKS_THRESH:
                total_valid_dt_count += 1

    # added for recall
    if len(dts) == 0:
        FN = len(gts)

    if total_valid_dt_count > len(gts):
        FP = total_valid_dt_count - len(gts)
        TP = len(gts)
    else:
        TP = total_valid_dt_count
        FP = 0
        FN = len(gts) - total_valid_dt_count

    return [TP, FP, FN]


def get_keypoints_oks_metrics_full(
    target_keypoint_dict, heatmaps_output, mid_offsets
):
    keypoints_target = {}
    keypoints_output = {}
    target_keypoint_list = target_keypoint_dict["keypoints"]
    summary_dict = {}
    output_keypoints_dict = get_keypoints(heatmaps_output)
    pred_skels = group_skeletons(output_keypoints_dict, mid_offsets)
    output_keypoints_list = [
        skel for skel in pred_skels if (skel[:, 2] > 0).sum() > 4
    ]
    tp_fp_list = computeOks_full(
        target_keypoint_dict, target_keypoint_list, output_keypoints_list
    )

    return tp_fp_list


def computepck_full(target_keypoint_dict, gts, dts):

    if len(gts) == 0:
        # no gt values for image, need to omit in metrics
        return [-1, -1, -1]

    total_valid_gt_count = 0
    total_valid_dt_count = 0
    total_fp_dt_count = 0
    total_tp_dt_count = 0
    FN = 0

    pck = np.zeros((len(dts), len(gts)))

    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):

        e = np.array([9999])  # high initial value for not passing pck thresh
        g = np.array(gt)
        xg = g[:, 0]
        yg = g[:, 1]
        vg = g[:, 2]
        k1 = np.count_nonzero(vg > 0)
        total_valid_gt_count += k1
        # torso diameter- distance between left shoulder and right hip of each ground-truth pose
        torso_dia = config.get_torso_dia(g, input_shape)

        for i, dt in enumerate(dts):
            d = np.array(dt)
            xd = d[:, 0]
            yd = d[:, 1]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
                # added
                e = (dx**2 + dy**2) ** 0.5

                e = e[vg > 0]

            total_valid_dt_count += e[e < torso_dia].shape[0]

    # added for recall
    if len(dts) == 0:
        FN = total_valid_gt_count

    if total_valid_dt_count > total_valid_gt_count:
        FP = total_valid_dt_count - total_valid_gt_count
        TP = total_valid_gt_count
    else:
        TP = total_valid_dt_count
        FP = 0
        FN = total_valid_gt_count - total_valid_dt_count

    return [TP, FP, FN]


def get_keypoints_pck_metrics_full(
    target_keypoint_dict, heatmaps_output, mid_offsets
):
    keypoints_target = {}
    keypoints_output = {}
    target_keypoint_list = target_keypoint_dict["keypoints"]
    torso_dia_list = []
    output_keypoints_dict = get_keypoints(heatmaps_output)
    pred_skels = group_skeletons(output_keypoints_dict, mid_offsets)
    output_keypoints_list = [
        skel for skel in pred_skels if (skel[:, 2] > 0).sum() > 4
    ]
    tp_fp_list = computepck_full(
        target_keypoint_dict, target_keypoint_list, output_keypoints_list
    )
    return tp_fp_list


## THIS IS THE ALGORITHM DESCRIBED IN THE PAPER:

# def group_skeletons(keypoints, mid_offsets, heatmaps):
#     keypoints.sort(key=(lambda kp: kp['conf']), reverse=True)
#     skeletons = []
#     dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

#     skeleton_graph = {i:[] for i in range(config.NUM_KP)}
#     for i in range(config.NUM_KP):
#         for j in range(config.NUM_KP):
#             if (i,j) in config.EDGES or (j,i) in config.EDGES:
#                 skeleton_graph[i].append(j)
#                 skeleton_graph[j].append(i)

#     for kp in keypoints:
#         if any([np.linalg.norm(kp['xy']-s[kp['id'], :2]) <= 4 for s in skeletons]):
#             continue
#         this_skel = np.zeros((config.NUM_KP, 3))
#         this_skel[kp['id'], :2] = kp['xy']
#         this_skel[kp['id'], 2] = heatmaps[int(kp['xy'][1]), int(kp['xy'][0]), kp['id']]
#         path = iterative_bfs(skeleton_graph, kp['id'])[1:]
#         for edge in path:
#             if this_skel[edge[0],2] == 0:
#                 continue
#             mid_idx = dir_edges.index(edge)
#             offsets = mid_offsets[:,:,2*mid_idx:2*mid_idx+2]
#             from_kp = tuple(this_skel[edge[0],:2].astype('int32'))
#             this_skel[edge[1],:2] = this_skel[edge[0],:2] + offsets[from_kp[1], from_kp[0], :]
#             this_skel[edge[1], 2] = heatmaps[int(this_skel[edge[1],1]), int(this_skel[edge[1],0]), edge[1]]

#         skeletons.append(this_skel)

#     return skeletons


def group_skeletons(keypoints, mid_offsets):
    keypoints.sort(key=(lambda kp: kp["conf"]), reverse=True)
    skeletons = []
    dir_edges = config.EDGES + [edge[::-1] for edge in config.EDGES]

    skeleton_graph = {i: [] for i in range(config.NUM_KP)}
    for i in range(config.NUM_KP):
        for j in range(config.NUM_KP):
            if (i, j) in config.EDGES or (j, i) in config.EDGES:
                skeleton_graph[i].append(j)
                skeleton_graph[j].append(i)

    while len(keypoints) > 0:
        kp = keypoints.pop(0)
        if any(
            [
                np.linalg.norm(kp["xy"] - s[kp["id"], :2]) <= 10
                for s in skeletons
            ]
        ):
            continue
        this_skel = np.zeros((config.NUM_KP, 3))
        this_skel[kp["id"], :2] = kp["xy"]
        this_skel[kp["id"], 2] = kp["conf"]
        path = iterative_bfs(skeleton_graph, kp["id"])[1:]
        for edge in path:
            if this_skel[edge[0], 2] == 0:
                continue
            mid_idx = dir_edges.index(edge)
            offsets = mid_offsets[:, :, 2 * mid_idx : 2 * mid_idx + 2]
            from_kp = tuple(np.round(this_skel[edge[0], :2]).astype("int32"))
            proposal = (
                this_skel[edge[0], :2] + offsets[from_kp[1], from_kp[0], :]
            )
            matches = [
                (i, keypoints[i])
                for i in range(len(keypoints))
                if keypoints[i]["id"] == edge[1]
            ]
            matches = [
                match
                for match in matches
                if np.linalg.norm(proposal - match[1]["xy"])
                <= config.NMS_THRESH
            ]
            if len(matches) == 0:
                continue
            matches.sort(key=lambda m: np.linalg.norm(m[1]["xy"] - proposal))
            to_kp = np.round(matches[0][1]["xy"]).astype("int32")
            to_kp_conf = matches[0][1]["conf"]
            keypoints.pop(matches[0][0])
            this_skel[edge[1], :2] = to_kp
            this_skel[edge[1], 2] = to_kp_conf

        skeletons.append(this_skel)

    return skeletons


def get_instance_masks(skeletons, seg_mask, long_offsets, threshold=True):
    map_shape = seg_mask.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1, 0, 2))
    features = np.tile(idx, config.NUM_KP) + long_offsets
    num_skels = len(skeletons)

    p_i, p_j = np.nonzero(seg_mask > 0.5)
    n = len(p_i)
    probs = np.zeros((n, num_skels))

    for j in range(num_skels):
        scale = (skeletons[j].max(axis=0) - skeletons[j].min(axis=0))[
            :2
        ].prod()
        scale = np.sqrt(scale)
        this_prob = np.zeros((n,))
        norm_factor = 0.0
        for k in range(config.NUM_KP):
            if skeletons[j][k, 2] == 0:
                continue
            dists = features[p_i, p_j, 2 * k : 2 * k + 2] - np.array(
                [[skeletons[j][k, 0], skeletons[j][k, 1]]]
            )
            p = np.sqrt(np.square(dists).sum(axis=-1))
            p *= skeletons[j][k, 2] / scale
            this_prob += p
            norm_factor += skeletons[j][k, 2]
        probs[:, j] = this_prob / norm_factor

    P = 1000.0 * np.ones(map_shape + (num_skels,))
    P[p_i, p_j, :] = probs

    masks = np.zeros(map_shape + (num_skels,))
    masks[
        idx[:, :, 1].flatten(),
        idx[:, :, 0].flatten(),
        P.argmin(axis=-1).flatten(),
    ] = 1
    # This should really use the instance mask distance threshold,
    # but for now is just assigns all pixels in the seg mask to an instance
    # mask for which the metric distance is lowest
    if threshold:
        masks[P.min(axis=-1) > config.INSTANCE_SEG_THRESH, :] = 0
    else:
        masks[P.min(axis=-1) > 999.0, :] = 0
    return [np.squeeze(m) for m in np.split(masks, num_skels, axis=-1)]


def get_skeletons_and_masks(outputs):
    kp_maps, short_offsets, mid_offsets, long_offsets, seg_mask = outputs
    heatmaps = compute_heatmaps(kp_maps, short_offsets)
    for i in range(config.NUM_KP):
        heatmaps[:, :, i] = gaussian_filter(heatmaps[:, :, i], sigma=2)
    pred_kp = get_keypoints(heatmaps)
    skeletons = group_skeletons(pred_kp, mid_offsets, kp_maps)
    instance_masks = get_instance_masks(skeletons, seg_mask, long_offsets)

    return skeletons, instance_masks
