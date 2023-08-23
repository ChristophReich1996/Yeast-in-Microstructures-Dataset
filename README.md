# An Instance Segmentation Dataset of Yeast Cells in Microstructures

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2304.07597-B31B1B.svg)](https://arxiv.org/abs/2304.07597)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Christoph Reich](https://christophreich1996.github.io)
, [Tim Prangemeier](https://scholar.google.com/citations?user=Ut23u2YAAAAJ&hl=en)
, [André O. Françani](https://scholar.google.com/citations?user=031ZSLQAAAAJ&hl=en)
& [Heinz Koeppl](https://www.bcs.tu-darmstadt.de/team_sos/koepplheinz_sos.en.jsp)**<br/>

## | [Project Page](https://christophreich1996.github.io/yeast_in_microstructures_dataset/) | [Paper](https://arxiv.org/abs/2304.07597) |  [Download Dataset](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/3799) |

<p align="center">
  <img src="github/first_fig.gif"  alt="1" width = 600px height = 300px >
</p>

<p align="center">
  This repository includes the <b>official</b> and <b>maintained</b> <a href="https://pytorch.org/">PyTorch</a> <b>validation</b> (+ data loading & visualization) code of the Yeast in Microstructures dataset proposed in <a href="https://arxiv.org">An Instance Segmentation Dataset of Yeast Cells in Microstructures</a>.
</p>

```
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/3799/yeast_cell_in_microstructures_dataset.zip
```

## Abstract

*Extracting single-cell information from microscopy data requires accurate instance-wise segmentations. Obtaining
pixel-wise segmentations from microscopy imagery remains a challenging task, especially with the added complexity of
microstructured environments. This paper presents a novel dataset for segmenting yeast cells in microstructures. We
offer pixel-wise instance segmentation labels for both cells and trap microstructures. In total, we release 493 densely
annotated microscopy images. To facilitate a unified comparison between novel segmentation algorithms, we propose a
standardized evaluation strategy for our dataset. The aim of the dataset and evaluation strategy is to facilitate the
development of new cell segmentation approaches.*

**If you use our dataset or find this research useful in your work, please cite our paper:**

```bibtex
@inproceedings{Reich2023,
        title={{An Instance Segmentation Dataset of Yeast Cells in Microstructures}},
        author={Reich, Christoph and Prangemeier, Tim and Fran{\c{c}}ani, Andr{\'e} O and Koeppl, Heinz},
        booktitle={{International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)}},
        year={2023},
        organization={IEEE}
}
```

## Table of Contents
1. [Installation](#installation)
2. [Dataformat](#dataformat)
3. [Dataset Class](#dataset-class)
4. [Evaluation](#evaluation)
5. [Visualization](#visualization)
6. [Additional Unlabeled Data](#additional-unlabeled-data)
7. [Acknowledgements](#acknowledgements)

## Installation

The validation, data loading, and visualization code can be installed as a Python package by running:

```shell script
pip install git+https://github.com/ChristophReich1996/Yeast-in-Microstructures-Dataset.git
```

All dependencies are listed in [requirements.txt](requirements.txt).

## Dataformat

The dataset is split into a training, validation, and test set. Please refer to the paper for more information on this.

```
├── test
│     ├── bounding_boxes
│     ├── classes
│     ├── inputs
│     └── instances
├── train
│     ├── bounding_boxes
│     ├── classes
│     ├── inputs
│     └── instances
└── val
      ├── bounding_boxes
      ├── classes
      ├── inputs
      └── instances
```

Every subset (train, val, and test) includes four different folders (`inputs`, `instances`, `classes`, `bounding_boxes`)
. The `inputs` folder includes the input images each with the shape `[128, 128]`. The `instances` folder holds the instance
maps of a shape of `[N, 128, 128]` (`N` is the number of instances). The `classes` holds the semantic class information
of each instance as a tensor of shape `[N]`. The `bounding_boxes` folder offers axis-aligned bounding boxes for each
instance of shape `[N, 4 (x0y0x1y1)]`. Every sample of the dataset has a `.pt` file in each of the four folders.
The `.pt` file can directly be loaded as a PyTorch Tensor
with [`torch.load(...)`](https://pytorch.org/docs/stable/generated/torch.load.html). For details on the data loading
please have a look at the [dataset class implementation](yim_dataset/data/dataset.py).

## Dataset Class<a name="dataset-class" />

This repo includes a PyTorch dataset class implementation (in the `yim_dataset.data` module) of the Yeast in
Microstructures dataset, located in the module `yim_dataset.data`. The dataset class implementation loads the dataset
and returns the images, instance maps, bounding boxes, and semantic classes.

```python
import yim_dataset
from torch import Tensor
from torch.utils.data import Dataset

# Init dataset
dataset: Dataset = yim_dataset.data.YIMDataset(path="/some_path_to_data/train", return_absolute_bounding_box=False)
# Get first sample of the dataset
image, instances, bounding_boxes, class_labels = dataset[0]  # type: Tensor, Tensor, Tensor, Tensor
# Show shapes
print(image.shape)  # [1, 256, 256]
print(instances.shape)  # [N, 256, 256]
print(bounding_boxes.shape)  # [N, 4 (xcycwh, relative format)]
print(class_labels)  # [N, C=2 (trap=0 and cell=1)]
```

The dataset class implementation also offers support for
custom [Kornia data augmentations](https://kornia.readthedocs.io/en/latest/applications/image_augmentations.html). You
can pass an [AugmentationSequential](https://kornia.readthedocs.io/en/latest/augmentation.container.html) object to the
dataset class. The following example utilizes random horizontal and vertical flipping as well as random Gaussian blur
augmentations.

```python
import kornia.augmentation
import yim_dataset
from torch.utils.data import Dataset

# Init augmentations
augmentations = kornia.augmentation.AugmentationSequential(
    kornia.augmentation.RandomHorizontalFlip(p=0.5),
    kornia.augmentation.RandomVerticalFlip(p=0.5),
    kornia.augmentation.RandomGaussianBlur(kernel_size=(31, 31), sigma=(9, 9), p=0.5),
    data_keys=["input", "bbox_xyxy", "mask"],
    same_on_batch=False,
)
# Init dataset
dataset: Dataset = yim_dataset.data.YIMDataset(path="/some_path_to_data/train", augmentations=augmentations)
```

**Note that it is necessary to pass `["input", "bbox_xyxy", "mask"]` as data keys!** If a different data key
configuration is given a runtime error is raised.

For wrapping the dataset with the PyTorch DataLoader please use
the [custom collide function](yim_dataset/data/dataset.py).

```python
from typing import List

import yim_dataset
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# Init dataset
dataset: Dataset = yim_dataset.data.YIMDataset(path="/some_path_to_data/train", return_absolute_bounding_box=False)
data_loader = DataLoader(
    dataset=dataset,
    num_workers=2,
    batch_size=2,
    drop_last=True,
    collate_fn=yim_dataset.data.collate_function_yim_dataset,
)
# Get a sample from dataloader
images, instances, bounding_boxes, class_labels = next(
    iter(data_loader))  # type: Tensor, List[Tensor], List[Tensor], List[Tensor]
# Show shapes
print(images.shape)  # [B, 1, 256, 256]
print(instances.shape)  # list([N, 256, 256])
print(bounding_boxes.shape)  # list([N, 4 (xcycwh, relative format)])
print(class_labels)  # list([N, C=2 (trap=0 and cell=1)])
```

<details>
  <summary>All Dataset Class Parameters</summary>

[YIMDataset](yim_dataset/data/dataset.py) parameters:

| Parameter                                            | Default value                   | Info                                                               |
|------------------------------------------------------|---------------------------------|--------------------------------------------------------------------|
| `path: str`                                          | -                               | Path to dataset as a string.                                       |
| `augmentations: Optional[AugmentationSequential]`    | `None`                          | Augmentations to be used. If `None` no augmentation is employed.   |
| `normalize: bool`                                    | `True`                          | If true images are normalized by the given normalization function. |
| `normalization_function: Callable[[Tensor], Tensor]` | `normalize` (0 mean, unit std.) | Normalization function.                                            |
| `return_absolute_bounding_box: bool`                 | `False`                         | If true BBs returned absolut format (else relative)                |

</details>

We provide a full dataset and data loader example in [`example_eval.py`](example_data.py).

If this dataset class implementation is not sufficient for your application please customize the existing code or open a
pull request with extending the existing implementation.

## Evaluation

We propose to validate segmentation predictions on our dataset by using
the [Panoptic Quality](https://arxiv.org/abs/1801.00868) and the cell class IoU. We implement both metrics as a
TorchMetrics metric in the `yim_dataset.eval` module. Both metrics ([PanopticQuality](yim_dataset/eval/pq.py)
and [CellIoU](yim_dataset/eval/cell_iou.py)) can be used like all TorchMetrics metrics. The input to both metrics is the
prediction, composed of the instance maps (list of tensors) and semantic class prediction (list of tensors), and the
label is also composed of instance maps and semantic classes. Note that the instance maps are not allowed to overlap.
Additionally, both metrics assume thresholded instance maps and hard semantic classes (no logits).

```python
import yim_dataset
from torchmetrics import Metric

pq: Metric = yim_dataset.eval.PanopticQuality()
cell_iou: Metric = yim_dataset.eval.CellIoU()

for index, (images, instances, bounding_boxes, class_labels) in enumerate(data_loader):
    # Make prediction
    instances_pred, bounding_boxes_pred, class_labels_pred = model(
        images)  # type: List[Tensor], List[Tensor], List[Tensor]
    # Get semantic classes form one-hot vector
    class_labels = [c.argmax(dim=-1) for c in class_labels]
    class_labels_pred = [c.argmax(dim=-1) for c in class_labels_pred]
    # Compute metrics
    pq.update(
        instances_pred=instances_pred,
        classes_pred=class_labels_pred,
        instances_target=instances,
        classes_target=class_labels,
    )
    cell_iou.update(
        instances_pred=instances_pred,
        classes_pred=class_labels_pred,
        instances_target=instances,
        classes_target=class_labels,
    )
# Compute final metric
print(f"Panoptic Quality: {pq.compute().item()}")
print(f"Cell class IoU: {cell_iou.compute().item()}")
```

A full working example is provided in [`example_eval.py`](example_eval.py).

## Visualization

This implementation (`yim_dataset.vis` module) also includes various functions for reproducing the plots from the paper.
The instance segmentation overlay (image + instance maps + BB + classes), as shown at the top, can be achieved by:

```python
import yim_dataset
from torch import Tensor
from torch.utils.data import Dataset

# Init dataset
dataset: Dataset = yim_dataset.data.YIMDataset(path="/some_path_to_data/train", return_absolute_bounding_box=False)
# Get first sample of the dataset
image, instances, bounding_boxes, class_labels = dataset[0]  # type: Tensor, Tensor, Tensor, Tensor
# Plot 
yim_dataset.vis.plot_image_instances_bb_classes(
    image=image,
    instances=instances,
    bounding_boxes=yim_dataset.data.bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes),
    class_labels=class_labels.argmax(dim=1),
    save=False,
    show=True,
    show_class_label=True,
)
```

All plot functions entail the parameter `show: bool` and `save: bool`. If `show=True` the plot is directly visualized by
calling `plt.show()`. If you want to save the plot to a file set `save=True` and provide the path and file
name (`file_path: str`).

An example use of all visualization functions is provided in [`example_vis.py`](example_vis.py).

## Additional Unlabeled Data<a name="additional-unlabeled-data" />

**Note that there are also additional unlabeled data available from the same domain.** In the
paper [Multi-StyleGAN: Towards Image-Based Simulation of Time-Lapse Live-Cell](https://christophreich1996.github.io/multi_stylegan/)
we proposed an unlabeled dataset of ~9k images (sequences) of yeast cells in microstructures. **The dataset is available at
[TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2880).** Please cite the following paper if you are
using the unlabeled images in your research:

```bibtex
@inproceedings{Reich2021,
        title={{Multi-StyleGAN: Towards Image-Based Simulation of Time-Lapse Live-Cell Microscopy}},
        author={Reich, Christoph and Prangemeier, Tim and Wildner, Christian and Koeppl, Heinz},
        booktitle={{International Conference on Medical image computing and computer-assisted intervention (MICCAI)}},
        year={2021},
        organization={Springer}
}
```

## Acknowledgements

We thank [Christoph Hoog Antink](https://www.etit.tu-darmstadt.de/kismed/team_kismed/hoog_antink.de.jsp) for insightful
discussions, [Klaus-Dieter Voss](https://www.bcs.tu-darmstadt.de/team_sos/mitarbeiterliste_sos.en.jsp) for aid with the
microfluidics fabrication, [Jan Basrawi](https://www.linkedin.com/in/jan-basrawi-b90117144/) for contributing to data
labeling, and [Robert Sauerborn](https://github.com/R98) for aid with setting up the project page.

Credit to [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Lightning AI)
, [Kornia](https://github.com/kornia/kornia), and [PyTorch](https://github.com/pytorch/pytorch) for providing the basis
of this implementation.

This work was supported by the Landesoffensive für wissenschaftliche Exzellenz as part of the LOEWE Schwerpunkt
CompuGene. H.K. acknowledges the support from the European Research Council (ERC) with the consolidator grant CONSYN (
nr. 773196). C.R. acknowledges the support of NEC Laboratories America, Inc.
