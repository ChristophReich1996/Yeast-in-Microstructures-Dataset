from typing import List

import torch
import torchmetrics
from torch import Tensor


class CellIoU(torchmetrics.Metric):
    """This class implements the cell class IoU for the YIM dataset format."""

    def __init__(self, cell_class_index: int = 1) -> None:
        """Constructor method.

        Args:
            cell_class_index (int): Cell class index. Default 1.
        """
        # Call super constructor
        super(CellIoU, self).__init__()
        # Save parameter
        self.cell_class_index: int = cell_class_index
        # Init states
        self.add_state("intersection", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        instances_pred: List[Tensor],
        classes_pred: List[Tensor],
        instances_target: List[Tensor],
        classes_target: List[Tensor],
    ) -> None:
        """Updates the state of the metrix with a new sample.

        Notes:
            instances_pred must be a binary map without overlapping instances.
            classes_pred must entail the semantic class not the logit vector.
            If no instance was detected the respective list entry should be None!
            We assume that at least a single object is present in the label!
            We also assume that the spatial dimensions between the label and the prediction are matching.

        Args:
            instances_pred (List[Tensor]): List of instance masks each of shape [N, H, W].
            classes_pred (List[Tensor]): List of semantic classes each of the shape [N].
            instances_target (List[Tensor]): List of instance mask labels each of shape [N, H, W].
            classes_target (List[Tensor]): List of semantic class labels each of the shape [N].
        """
        # Make semantic label
        semantic_target: Tensor = torch.zeros(
            len(classes_target), instances_target[0].shape[-2], instances_target[0].shape[-1]
        )
        for index, classes in enumerate(classes_target):
            semantic_target[index] = (classes.view(-1, 1, 1) * instances_target[index]).sum(dim=0)
        # Make semantic prediction
        semantic_pred: Tensor = torch.zeros(
            len(classes_pred), instances_target[0].shape[-2], instances_target[0].shape[-1]
        )
        for index, classes in enumerate(classes_pred):
            if classes is not None:
                semantic_pred[index] = (classes.view(-1, 1, 1) * instances_pred[index]).sum(dim=0)
        # Get semantic cell maps
        semantic_cell_target: Tensor = semantic_target == self.cell_class_index
        semantic_cell_pred: Tensor = semantic_pred == self.cell_class_index
        # Compute intersection and union
        self.intersection += torch.logical_and(semantic_cell_target, semantic_cell_pred).sum()
        self.union += torch.logical_or(semantic_cell_target, semantic_cell_pred).sum()

    def compute(self) -> Tensor:
        """Method computes the final metric.

        Returns:
            cell_iou (Tensor): Cell IoU metric.
        """
        # Compute cell class IoU
        cell_iou: Tensor = self.intersection / self.union.clip(min=1e-06)
        return cell_iou
