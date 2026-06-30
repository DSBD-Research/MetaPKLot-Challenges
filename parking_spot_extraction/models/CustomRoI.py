# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

from mmdet.registry import MODELS
from torch import Tensor
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList
from mmdet.models.utils import empty_instances
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
import torch


@MODELS.register_module()
class CustomRoI(StandardRoIHead):
    def _mask_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor = None,
        pos_inds: Optional[Tensor] = None,
        bbox_feats: Optional[Tensor] = None,
    ) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            pos_inds (Tensor, optional): Indices of positive samples.
                Defaults to None.
            bbox_feats (Tensor): Extract bbox RoI features. Defaults to None.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `mask_preds` (Tensor): Mask prediction.
                - `mask_feats` (Tensor): Extract mask RoI features.
        """
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[: self.mask_roi_extractor.num_inputs], rois
            )
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds = self.mask_head(mask_feats)
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats)
        return mask_results

    # def predict_mask(self,
    #                  x: Tuple[Tensor],
    #                  batch_img_metas: List[dict],
    #                  results_list: InstanceList,
    #                  rescale: bool = False) -> InstanceList:
    #     """Perform forward propagation of the mask head and predict detection
    #     results on the features of the upstream network.

    #     Args:
    #         x (tuple[Tensor]): Feature maps of all scale level.
    #         batch_img_metas (list[dict]): List of image information.
    #         results_list (list[:obj:`InstanceData`]): Detection results of
    #             each image.
    #         rescale (bool): If True, return boxes in original image space.
    #             Defaults to False.

    #     Returns:
    #         list[:obj:`InstanceData`]: Detection results of each image
    #         after the post process.
    #         Each item usually contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #             - masks (Tensor): Has a shape (num_instances, H, W).
    #     """
    #     # don't need to consider aug_test.
    #     bboxes = [res.bboxes for res in results_list]
    #     mask_rois = bbox2roi(bboxes)
    #     if mask_rois.shape[0] == 0:
    #         results_list = empty_instances(
    #             batch_img_metas,
    #             mask_rois.device,
    #             task_type='mask',
    #             instance_results=results_list,
    #             mask_thr_binary=self.test_cfg.mask_thr_binary)
    #         return results_list

    #     mask_results = self._mask_forward(x, mask_rois)
    #     mask_preds = mask_results['mask_preds']
    #     # split batch mask prediction back to each image
    #     num_mask_rois_per_img = [len(res) for res in results_list]
    #     mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

    #     # TODO: Handle the case where rescale is false
    #     results_list = self.mask_head.predict_by_feat(
    #         mask_preds=mask_preds,
    #         results_list=results_list,
    #         batch_img_metas=batch_img_metas,
    #         rcnn_test_cfg=self.test_cfg,
    #         rescale=rescale)

    #     results_list[0]['mask_feats'] = mask_results['mask_feats'].mean(dim=[2, 3])
    #     return results_list

    @torch.no_grad()
    def _compute_image_embeddings(self, x: Tuple[Tensor]) -> Tensor:
        """Global image embeddings by GAP over FPN levels and concatenation.

        Args:
            x (tuple[Tensor]): FPN features (each [B, C, H, W])

        Returns:
            Tensor: Per-image embeddings of shape [B, sum(C_l)]
        """
        # Use the same feature levels consumed by the mask ROI extractor
        num_inputs = getattr(self.mask_roi_extractor, "num_inputs", len(x))
        pooled_per_level = []
        for feat in x[:num_inputs]:
            # Global average pool over H, W: [B, C, H, W] -> [B, C]
            pooled = feat.mean(dim=(2, 3))
            pooled_per_level.append(pooled)
        img_embs = torch.cat(pooled_per_level, dim=1)  # [B, sum(C_l)]
        # Optional: L2-normalize (comment out if you prefer raw)
        img_embs = torch.nn.functional.normalize(img_embs, dim=1)
        return img_embs

    def predict_mask(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        results_list: InstanceList,
        rescale: bool = False,
    ) -> InstanceList:

        # ---- 1) Compute per-image embeddings from backbone/neck features
        # (one embedding per image in the batch)
        img_embs = self._compute_image_embeddings(x)  # [B, D]
        assert img_embs.size(0) == len(
            results_list
        ), f"Batch size {img_embs.size(0)} != results_list length {len(results_list)}"

        # ---- 2) Regular mask prediction path
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)

        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )
            # Attach image embeddings even when no RoIs are present
            # for i, res in enumerate(results_list):
            #     res.set_field("img_emb", img_embs[i])
            #     # Provide an empty mask_feats tensor for consistency
            #     # res.set_field("mask_feats", img_embs.new_zeros((0, 0)))
            return results_list

        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results[
            "mask_preds"
        ]  # [sum_rois, num_classes, ph, pw] or similar
        mask_roi_feats = mask_results["mask_feats"]  # [sum_rois, C, ph, pw]

        # Split back per image
        num_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_rois_per_img, dim=0)

        # ---- 3) Finish mask post-processing (MMDet’s built-in)
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )

        # ---- 4) Attach per-image fields:
        # 4a) Per-image global embeddings
        for i, res in enumerate(results_list):
            # InstanceData allows arbitrary fields via attribute or set_field
            # res.set_field("img_emb", img_embs[i])  # shape [D]
            # print(f"Image z{i} embedding shape: {img_embs[i].shape}")
            # res["img_emb"] = img_embs[i]  # alternative syntax
            res.set_metainfo({"img_emb": img_embs[i]})

        # 4b) Per-instance mask embeddings (pooled RoI features)
        #     Pool spatial dims of the RoI-aligned mask features and split per image
        mask_roi_feats_pooled = mask_roi_feats.mean(dim=(2, 3))  # [sum_rois, C]
        mask_roi_feats_pooled = mask_roi_feats_pooled.split(num_rois_per_img, dim=0)

        for i, res in enumerate(results_list):
            # Tensor of shape [num_instances_in_img, C]
            res["mask_feats"] = mask_roi_feats_pooled[i]
            # results_list[i]['mask_feats'] = mask_results['mask_feats'].mean(dim=[2, 3])

        return results_list
