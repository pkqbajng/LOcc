import numpy as np

class Metric_SSC():
    def __init__(self,
                 ignore_label=[0, 17],
                 class_names=['others','barrier','bicycle','bus','car',
                            'construction_vehicle','motorcycle','pedestrian',
                            'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free'],
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.ignore_label = ignore_label
        self.class_names = class_names
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.reset()
    
    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        # k = (gt >= self.ignore_label) & (gt < n_cl)  # exclude 255
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar=None, mask_camera=None):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera.astype(np.bool)]
            masked_semantics_pred = semantics_pred[mask_camera.astype(np.bool)]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar.astype(np.bool)]
            masked_semantics_pred = semantics_pred[mask_lidar.astype(np.bool)]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
        
        batch_hist, _, _ = self.hist_info(self.num_classes, masked_semantics_pred, masked_semantics_gt)
        self.hist = self.hist + batch_hist

    def count_miou(self):
        miou = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist)+ 1e-6)*100.0
        completion_tp = np.sum(self.hist[:-1, :-1])
        completion_fp = np.sum(self.hist[-1, :-1])
        completion_fn = np.sum(self.hist[:-1, -1])

        if completion_tp != 0:
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / (completion_tp + completion_fp + completion_fn)*100.0
        else:
            precision, recall, iou = 0, 0, 0

        iou_ssc = miou[:self.num_classes-1]  # exclude the empty voxel

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "iou_ssc": iou_ssc,  # class IOU
            "miou": np.nanmean(iou_ssc[:self.num_classes-1]),
            "class_names": self.class_names
        }