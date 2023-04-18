import torch
from torch import Tensor


def to_one_hot(
    input: Tensor,
    num_classes: int,
) -> Tensor:
    """Class input to one-hot tensor.

    Args:
        input (Tensor): Class number tensor of any shape.
        num_classes (int): Number of classes.

    Returns:
        one_hot (Tensor): One-hot tensor of the shape [(input shape), C]
    """
    one_hot = torch.zeros([input.shape[0], num_classes], dtype=torch.float)
    one_hot.scatter_(1, input.view(-1, 1).long(), 1)
    return one_hot


def bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes: Tensor) -> Tensor:
    """Converts a tensor of bounding boxes from the xcycwh format to the x0y0x1y1 format.

    Args:
        bounding_boxes (Tensor): Bounding box of shape [*, 4 (x center, y center, width, height)]

    Returns:
       bounding_box_converted (Tensor): Converted BBs of shape [*, 4 (x0, y0, x1, y1)]
    """
    x_center, y_center, width, height = bounding_boxes.unbind(dim=-1)
    bounding_box_converted = [
        (x_center - 0.5 * width),
        (y_center - 0.5 * height),
        (x_center + 0.5 * width),
        (y_center + 0.5 * height),
    ]
    return torch.stack(tensors=bounding_box_converted, dim=-1)


def bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes: Tensor) -> Tensor:
    """Converts a tensor of bounding boxes from the x0y0x1y1 format to the xcycwh format.

    Args:
        bounding_boxes (Tensor): Bounding box of shape [*, 4 (x0, y0, x1, y1)]

    Returns:
       bounding_box_converted (Tensor): Converted BBs of shape [*, 4 (x center, y center, width, height)]
    """
    x_0, y_0, x_1, y_1 = bounding_boxes.unbind(dim=-1)
    bounding_box_converted = [((x_0 + x_1) / 2), ((y_0 + y_1) / 2), (x_1 - x_0), (y_1 - y_0)]
    return torch.stack(tensors=bounding_box_converted, dim=-1)


def absolute_bounding_box_to_relative(
    bounding_boxes: Tensor,
    height: int,
    width: int,
    xcycwh: bool = False,
) -> Tensor:
    """Function converts an absolute bounding box to a relative one for a given image shape. Inplace operation!

    Args:
        bounding_boxes (Tensor): Bounding box with the format [*, 4]
        height (int): Height of the image.
        width (int): Width of the image.
        xcycwh (bool): Set to true if xcycwh format is given. Default False.

    Returns:
        bounding_boxes (Tensor): Relative bounding boxes of shape [*, 4 (x0, y0, x1, y1)]
    """
    # Case if xcycwh format is given
    if xcycwh:
        bounding_boxes = bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes)
    # Apply height and width
    bounding_boxes[..., [0, 2]] = bounding_boxes[..., [0, 2]] / width
    bounding_boxes[..., [1, 3]] = bounding_boxes[..., [1, 3]] / height
    # Return bounding box in the original format
    if xcycwh:
        return bounding_box_x0y0x1y1_to_xcycwh(bounding_boxes)
    return bounding_boxes


def normalize_0_1(input: Tensor) -> Tensor:
    """Normalizes a given tensor to a range of [0, 1].

    Args:
        input (Tensor): Input tensor of any shape.

    Returns:
        output (Tensor): Normalized output tensor of the same shape as the input.
    """
    # Perform normalization
    output: Tensor = (input - input.min()) / (input.max() - input.min())
    return output


def normalize(input: Tensor) -> Tensor:
    """Normalizes a given tensor to zero mean and unit standard deviation.

    Args:
        input (Tensor): Input tensor of any shape.

    Returns:
        output (Tensor): Normalized output tensor of the same shape as the input.
    """
    return (input - input.mean()) / input.std()
