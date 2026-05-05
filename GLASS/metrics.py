from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


'''PRO metric 변경'''

import numpy as np
from scipy.ndimage.measurements import label
from bisect import bisect

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def trapezoid(x, y, x_max=0.3):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x:     Samples from the domain of the function to integrate
               Need to be sorted in ascending order. May contain the same value
               multiple times. In that case, the order of the corresponding
               y values will affect the integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
               determined by interpolating between its neighbors. Must not lie
               outside of the range of x.

    Returns:
        Area under the curve.
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("""WARNING: Not all x and y values passed to trapezoid(...)
                 are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1]) #fps 0.3 전, 후 값으로 0.3일 때의 값 보간

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]
        
        x = rescale(x) #fps 0.0 ~ 0.3 -> 0.0 ~ 1.0

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction



def compute_pro(anomaly_maps, ground_truth_maps):
    """
    Compute the PRO curve for a set of anomaly maps with
    corresponding ground truth maps.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a
                           real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
                           contain binary-valued ground truth labels for each
                           pixel.
                           0 indicates that a pixel is anomaly-free.
                           1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """    
    
    ground_truth_maps[ground_truth_maps <= 0.45] = 0 #$ 원본: 0.45
    ground_truth_maps[ground_truth_maps > 0.45] = 1

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    # print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    
    '''강제 매칭 제거'''
    # zero = np.array([0.])
    # one = np.array([1.])

    # return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))
    

    pros = trapezoid(fprs, pros,x_max=0.3)
    
    return pros



# def compute_pro(masks, amaps, num_th=200):
#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
#     binary_amaps = np.zeros_like(amaps, dtype=bool)

#     min_th = amaps.min()
#     max_th = amaps.max()
#     delta = (max_th - min_th) / num_th

#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     for th in np.arange(min_th, max_th, delta):
#         binary_amaps[amaps <= th] = 0
#         binary_amaps[amaps > th] = 1

#         pros = []
#         for binary_amap, mask in zip(binary_amaps, masks):
#             binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
#             for region in measure.regionprops(measure.label(mask)):
#                 axes0_ids = region.coords[:, 0]
#                 axes1_ids = region.coords[:, 1]
#                 tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
#                 pros.append(tp_pixels / region.area)

#         inverse_masks = 1 - masks
#         fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
#         fpr = fp_pixels / inverse_masks.sum()

#         df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

#     df = df[df["fpr"] < 0.3]
#     df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

#     pro_auc = metrics.auc(df["fpr"], df["pro"])
#     return pro_auc
