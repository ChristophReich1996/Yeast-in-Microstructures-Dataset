from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torch import Tensor

from ..data.utils import normalize_0_1


def plot_image_instances_bb_classes(
    image: Tensor,
    instances: Tensor,
    bounding_boxes: Tensor,
    class_labels: Tensor,
    save: bool = False,
    show: bool = False,
    file_path: str = "plot.png",
    alpha: float = 0.3,
    show_class_label: bool = True,
    colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.05, 0.05, 0.05), (0.25, 0.25, 0.25)),
    cell_classes: Tuple[int, ...] = (1,),
    colors_cells: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.89019608),
        (1.0, 0.5, 0.90980392),
        (0.7, 0.0, 0.70980392),
        (0.7, 0.5, 0.73333333),
        (0.5, 0.0, 0.53333333),
        (0.5, 0.2, 0.55294118),
        (0.3, 0.0, 0.45),
        (0.3, 0.2, 0.45),
    ),
) -> None:
    """Plots the instance segmentation label overlaid with the microscopy image with bounding boxes.

    Args:
        image (Tensor): Input image of shape [3, H, W] or [1, H, W].
        instances (Tensor): Instances masks of shape [N, W, W].
        bounding_boxes (Tensor): Bounding boxes of shape [N, 4 (x1, y1, x2, y2)].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        alpha (float): Transparency factor of the instances. Default 0.3.
        show_class_label (bool): If true class label will be shown in plot. Default True.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_classes (Tuple[int, ...]): Tuple of cell classes. Default (1,).
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
    """
    # Normalize image to [0, 1]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    instances = instances.detach().cpu().numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = image.repeat(repeats=3, axis=-1)
    # Init counters
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Add instances to image
    for index, instance in enumerate(instances):
        # Case of cell instances
        if bool(class_labels[index] >= min(cell_classes)):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(
                    instance == 1,
                    image[:, :, c] * (1 - alpha)
                    + alpha * colors_cells[min(counter_cell_instance, len(colors_cells) - 1)][c],
                    image[:, :, c],
                )
            counter_cell_instance += 1
        # Case of trap class
        else:
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(
                    instance == 1,
                    image[:, :, c] * (1 - alpha)
                    + alpha * colors_traps[min(counter_trap_instance, len(colors_traps) - 1)][c],
                    image[:, :, c],
                )
            counter_trap_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Init counters
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] >= min(cell_classes)):
            rectangle = patches.Rectangle(
                (float(bounding_box[0]), float(bounding_box[1])),
                float(bounding_box[2]) - float(bounding_box[0]),
                float(bounding_box[3]) - float(bounding_box[1]),
                linewidth=3,
                edgecolor=colors_cells[min(counter_cell_instance, len(colors_cells) - 1)],
                facecolor="none",
                ls="dashed",
            )
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(
                    float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                    float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                    "Cell",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    color="white",
                    size=15,
                )
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            rectangle = patches.Rectangle(
                (float(bounding_box[0]), float(bounding_box[1])),
                float(bounding_box[2]) - float(bounding_box[0]),
                float(bounding_box[3]) - float(bounding_box[1]),
                linewidth=3,
                edgecolor=colors_traps[min(counter_trap_instance, len(colors_traps) - 1)],
                facecolor="none",
                ls="dashed",
            )
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(
                    float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                    float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                    "Trap",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    color="white",
                    size=15,
                )
            # Increment counter
            counter_trap_instance += 1
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches="tight", pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches="tight", pad_inches=0)
    # Close figure
    plt.close()


def plot_image_instances(
    image: Tensor,
    instances: Tensor,
    class_labels: Tensor,
    save: bool = False,
    show: bool = False,
    file_path: str = "plot.png",
    alpha: float = 0.5,
    colors_cells: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.89019608),
        (1.0, 0.5, 0.90980392),
        (0.7, 0.0, 0.70980392),
        (0.7, 0.5, 0.73333333),
        (0.5, 0.0, 0.53333333),
        (0.5, 0.2, 0.55294118),
        (0.3, 0.0, 0.45),
        (0.3, 0.2, 0.45),
    ),
    colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.05, 0.05, 0.05), (0.25, 0.25, 0.25)),
    cell_classes: Tuple[int, ...] = (1,),
) -> None:
    """Plots the instance segmentation label overlaid with the microscopy image.

    Args:
        image (Tensor): Input image of shape [3, H, W] or [1, H, W].
        instances (Tensor): Instances masks of shape [N, W, W].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        alpha (float): Transparency factor of the instances. Default 0.3.
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_classes (Tuple[int, ...]): Tuple of cell classes. Default (1,).
    """
    # Normalize image to [0, 1]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = image.repeat(repeats=3, axis=-1)
    # Init counters
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Add instances to image
    for index, instance in enumerate(instances):
        # Case of cell instances
        if bool(class_labels[index] >= min(cell_classes)):
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(
                    instance == 1,
                    image[:, :, c] * (1 - alpha)
                    + alpha * colors_cells[min(counter_cell_instance, len(colors_cells) - 1)][c],
                    image[:, :, c],
                )
            counter_cell_instance += 1
        # Case of trap class
        else:
            for c in range(image.shape[-1]):
                image[:, :, c] = np.where(
                    instance == 1,
                    image[:, :, c] * (1 - alpha)
                    + alpha * colors_traps[min(counter_trap_instance, len(colors_traps) - 1)][c],
                    image[:, :, c],
                )
            counter_trap_instance += 1

    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches="tight", pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches="tight", pad_inches=0)
    # Close figure
    plt.close()


def plot_instances_bb_classes(
    instances: Tensor,
    bounding_boxes: Tensor,
    class_labels: Tensor,
    save: bool = False,
    show: bool = False,
    file_path: str = "plot.png",
    colors_cells: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.89019608),
        (1.0, 0.5, 0.90980392),
        (0.7, 0.0, 0.70980392),
        (0.7, 0.5, 0.73333333),
        (0.5, 0.0, 0.53333333),
        (0.5, 0.2, 0.55294118),
        (0.3, 0.0, 0.45),
        (0.3, 0.2, 0.45),
    ),
    colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.3, 0.3, 0.3), (0.5, 0.5, 0.5)),
    cell_classes: Tuple[int, ...] = (1,),
    white_background: bool = False,
    show_class_label: bool = True,
) -> None:
    """Just plots the instance segmentation label with BB.

    Args:
        instances (Tensor): Instances masks of shape [N, W, W].
        bounding_boxes (Tensor): Bounding boxes of shape [N, 4 (x1, y1, x2, y2)].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_classes (Tuple[int, ...]): Tuple of cell classes. Default (1,).
        white_background (bool): If true a white background is utilized. Default False.
        show_class_label (bool): If true class name will be shown in the left bottom corner of each BB. Default True.
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init map to visualize instances
    instances_map = np.zeros((instances.shape[1], instances.shape[2], 3), dtype=np.float)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Instances to instances map
    for instance, class_label in zip(instances, class_labels):
        # Case if cell is present
        if bool(class_label >= min(cell_classes)):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(
                (1, 1, 3)
            ) * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_traps[min(counter_trap_instance, len(colors_cells) - 1)]).reshape(
                (1, 1, 3)
            ) * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_trap_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * instances_map.shape[0] / instances_map.shape[1])
    # Make background white if specified
    if white_background:
        for h in range(instances_map.shape[0]):
            for w in range(instances_map.shape[1]):
                if np.alltrue(instances_map[h, w, :] == np.array([0.0, 0.0, 0.0])):
                    instances_map[h, w, :] = np.array([1.0, 1.0, 1.0])
    # Plot image and instances
    ax.imshow(instances_map)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] >= min(cell_classes)):
            rectangle = patches.Rectangle(
                (float(bounding_box[0]), float(bounding_box[1])),
                float(bounding_box[2]) - float(bounding_box[0]),
                float(bounding_box[3]) - float(bounding_box[1]),
                linewidth=3,
                edgecolor=colors_cells[min(counter_cell_instance, len(colors_cells) - 1)],
                facecolor="none",
                ls="dashed",
            )
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(
                    float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                    float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                    "Cell",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    color="black" if white_background else "white",
                    size=15,
                )
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            rectangle = patches.Rectangle(
                (float(bounding_box[0]), float(bounding_box[1])),
                float(bounding_box[2]) - float(bounding_box[0]),
                float(bounding_box[3]) - float(bounding_box[1]),
                linewidth=3,
                edgecolor=colors_traps[min(counter_trap_instance, len(colors_traps) - 1)],
                facecolor="none",
                ls="dashed",
            )
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(
                    float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0]) - 2),
                    float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1]) - 2),
                    "Trap",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    color="black" if white_background else "white",
                    size=15,
                )
            # Increment counter
            counter_trap_instance += 1
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(
            file_path, dpi=instances_map.shape[1] * 4 / 3.845, transparent=True, bbox_inches="tight", pad_inches=0
        )
    # Show figure if utilized
    if show:
        plt.show(bbox_inches="tight", pad_inches=0)
    # Close figure
    plt.close()


def plot_instances(
    instances: Tensor,
    class_labels: Tensor,
    save: bool = False,
    show: bool = False,
    file_path: str = "",
    colors_cells: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.89019608),
        (1.0, 0.5, 0.90980392),
        (0.7, 0.0, 0.70980392),
        (0.7, 0.5, 0.73333333),
        (0.5, 0.0, 0.53333333),
        (0.5, 0.2, 0.55294118),
        (0.3, 0.0, 0.45),
        (0.3, 0.2, 0.45),
    ),
    colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.3, 0.3, 0.3), (0.5, 0.5, 0.5)),
    cell_classes: Tuple[int, ...] = (1,),
    white_background: bool = False,
) -> None:
    """Just plots the instance segmentation map.

    Args:
        instances (Tensor): Instances masks of shape [N, W, W].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_classes (Tuple[int, ...]): Tuple of cell classes. Default (1,).
        white_background (bool): If true a white background is utilized. Default False.
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init map to visualize instances
    instances_map = np.zeros((instances.shape[1], instances.shape[2], 3), dtype=np.float)
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Instances to instances map
    for instance, class_label in zip(instances, class_labels):
        # Case if cell is present
        if bool(class_label >= min(cell_classes)):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(
                (1, 1, 3)
            ) * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            # Add pixels of current instance, in the corresponding colour, to instances map
            instances_map += np.array(colors_traps[min(counter_trap_instance, len(colors_cells) - 1)]).reshape(
                (1, 1, 3)
            ) * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_trap_instance += 1
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * instances_map.shape[0] / instances_map.shape[1])
    # Make background white if specified
    if white_background:
        for h in range(instances_map.shape[0]):
            for w in range(instances_map.shape[1]):
                if np.alltrue(instances_map[h, w, :] == np.array([0.0, 0.0, 0.0])):
                    instances_map[h, w, :] = np.array([1.0, 1.0, 1.0])
    # Plot image and instances
    ax.imshow(instances_map)
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(
            file_path, dpi=instances_map.shape[1] * 4 / 3.845, transparent=True, bbox_inches="tight", pad_inches=0
        )
    # Show figure if utilized
    if show:
        plt.show(bbox_inches="tight", pad_inches=0)
    # Close figure
    plt.close()


def plot_image_bb_classes(
    image: Tensor,
    bounding_boxes: Tensor,
    class_labels: Tensor,
    save: bool = False,
    show: bool = False,
    file_path: str = "",
    show_class_label: bool = True,
    colors_cells: Tuple[Tuple[float, float, float], ...] = (1.0, 0.0, 0.89019608),
    colors_traps: Tuple[Tuple[float, float, float], ...] = (0.0, 0.0, 0.0),
    cell_classes: Tuple[int, ...] = (1,),
) -> None:
    """Plots the image with overlaid bounding boxes and classes.

    Args:
        image (Tensor): Input image of shape [3, height, width] or [1, height, width].
        bounding_boxes (Tensor): Bounding boxes of shape [N, 4 (x1, y1, x2, y2)].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        show_class_label (bool): If true class labels is shown in plot. Default True.
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_classes (Tuple[int, ...]): Tuple of cell classes. Default (1,).
    """
    # Normalize image to [0, 1]
    image = normalize_0_1(image)
    # Convert data to numpy
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Convert grayscale image to rgb
    if image.shape[-1] == 1:
        image = image.repeat(repeats=3, axis=-1)
    # Init figure
    fig, ax = plt.subplots()
    # Set size
    fig.set_size_inches(5, 5 * image.shape[0] / image.shape[1])
    # Plot image and instances
    ax.imshow(image)
    # Plot bounding_boxes and classes
    for index, bounding_box in enumerate(bounding_boxes):
        # Case if cell is present
        if bool(class_labels[index] >= min(cell_classes)):
            rectangle = patches.Rectangle(
                (float(bounding_box[0]), float(bounding_box[1])),
                float(bounding_box[2]) - float(bounding_box[0]),
                float(bounding_box[3]) - float(bounding_box[1]),
                linewidth=3,
                edgecolor=colors_cells,
                facecolor="none",
                ls="dashed",
            )
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(
                    float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0])) - 2,
                    float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1])) - 2,
                    "Cell",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    color="white",
                    size=15,
                )
        # Cas if trap is present
        else:
            rectangle = patches.Rectangle(
                (float(bounding_box[0]), float(bounding_box[1])),
                float(bounding_box[2]) - float(bounding_box[0]),
                float(bounding_box[3]) - float(bounding_box[1]),
                linewidth=3,
                edgecolor=colors_traps,
                facecolor="none",
                ls="dashed",
            )
            ax.add_patch(rectangle)
            if show_class_label:
                ax.text(
                    float(bounding_box[0]) + (float(bounding_box[2]) - float(bounding_box[0])) - 2,
                    float(bounding_box[1]) + (float(bounding_box[3]) - float(bounding_box[1])) - 2,
                    "Trap",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    color="white",
                    size=15,
                )
    # Axis off
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Save figure if utilized
    if save:
        plt.savefig(file_path, dpi=image.shape[1] * 4 / 3.845, transparent=True, bbox_inches="tight", pad_inches=0)
    # Show figure if utilized
    if show:
        plt.show(bbox_inches="tight", pad_inches=0)
    # Close figure
    plt.close()


def plot_individual_instances(
    instances: Tensor,
    class_labels: Tensor,
    save: bool = False,
    show: bool = False,
    file_path: str = "",
    colors_cells: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.89019608),
        (1.0, 0.5, 0.90980392),
        (0.7, 0.0, 0.70980392),
        (0.7, 0.5, 0.73333333),
        (0.5, 0.0, 0.53333333),
        (0.5, 0.2, 0.55294118),
        (0.3, 0.0, 0.45),
        (0.3, 0.2, 0.45),
    ),
    colors_traps: Tuple[Tuple[float, float, float], ...] = ((0.3, 0.3, 0.3), (0.5, 0.5, 0.5)),
    cell_classes: Tuple[int, ...] = (1,),
    white_background: bool = False,
) -> None:
    """Generates plots separate plots for every instance map.

    Args:
        instances (Tensor): Instances masks of shape [N, W, W].
        class_labels (Tensor): Class labels of each instance [N].
        save (bool): If true image will be stored under given path name. Default False.
        show (bool): If true plt.show() will be called. Default False.
        file_path (str): Path and name where image will be stored. Default "plot.png".
        colors_cells (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each trap instances.
        colors_traps (Tuple[Tuple[float, float, float], ...]): Tuple of RGB colors to visualize each cell instances.
        cell_classes (Tuple[int, ...]): Tuple of cell classes. Default (1,).
        white_background (bool): If true a white background is utilized. Default False.
    """
    # Convert data to numpy
    instances = instances.detach().cpu().numpy()
    class_labels = class_labels.detach().cpu().numpy()
    # Init counters to track the number of cells and traps for different colours
    counter_cell_instance = 0
    counter_trap_instance = 0
    # Instances to instances map
    for index, data in enumerate(zip(instances, class_labels)):
        # Unzip data
        instance, class_label = data
        # Case if cell is present
        if bool(class_label >= min(cell_classes)):
            # Add pixels of current instance, in the corresponding colour, to instances map
            instance = np.array(colors_cells[min(counter_cell_instance, len(colors_cells) - 1)]).reshape(
                (1, 1, 3)
            ) * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_cell_instance += 1
        # Cas if trap is present
        else:
            # Add pixels of current instance, in the corresponding colour, to instances map
            instance = np.array(colors_traps[min(counter_trap_instance, len(colors_cells) - 1)]).reshape(
                (1, 1, 3)
            ) * np.expand_dims(instance, axis=-1).repeat(3, axis=-1)
            # Increment counter
            counter_trap_instance += 1
        # Init figure
        fig, ax = plt.subplots()
        # Set size
        fig.set_size_inches(5, 5 * instance.shape[0] / instance.shape[1])
        # Make background white if specified
        if white_background:
            for h in range(instance.shape[0]):
                for w in range(instance.shape[1]):
                    if np.alltrue(instance[h, w, :] == np.array([0.0, 0.0, 0.0])):
                        instance[h, w, :] = np.array([1.0, 1.0, 1.0])
        # Plot image and instances
        ax.imshow(instance)
        # Axis off
        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Save figure if utilized
        if save:
            plt.savefig(
                file_path.replace(".", "_{}.".format(index)),
                dpi=instance.shape[1] * 4 / 3.845,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
            )
        # Show figure if utilized
        if show:
            plt.show(bbox_inches="tight", pad_inches=0)
        # Close figure
        plt.close()
