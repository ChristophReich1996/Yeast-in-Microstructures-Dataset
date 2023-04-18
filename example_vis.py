import kornia.augmentation

import yim_dataset

DATASET_PATH: str = "/Users/christoph/Desktop/yeast_cell_in_microstructures_dataset/dataset/train"


def main() -> None:
    # Init augmentations
    transforms = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomHorizontalFlip(p=1.0),
        kornia.augmentation.RandomVerticalFlip(p=1.0),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(31, 31), sigma=(9, 9), p=1.0),
        data_keys=["input", "bbox_xyxy", "mask"],
        same_on_batch=False,
    )
    # Init dataset
    dataset = yim_dataset.data.YIMDataset(path=DATASET_PATH, augmentations=transforms, return_absolute_bounding_box=True)
    # Get sample from dataset
    image, instances, bounding_boxes, class_labels = dataset[4]
    # Plotting
    yim_dataset.vis.plot_image_instances(
        image=image, instances=instances, class_labels=class_labels.argmax(dim=1), save=False, show=True
    )
    yim_dataset.vis.plot_instances(instances=instances, class_labels=class_labels.argmax(dim=1), save=False, show=True)
    yim_dataset.vis.plot_image_bb_classes(
        image=image,
        bounding_boxes=yim_dataset.data.bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes),
        class_labels=class_labels.argmax(dim=1),
        save=False,
        show=True,
        show_class_label=True,
    )
    yim_dataset.vis.plot_image_instances_bb_classes(
        image=image,
        instances=instances,
        bounding_boxes=yim_dataset.data.bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes),
        class_labels=class_labels.argmax(dim=1),
        save=False,
        show=True,
        show_class_label=True,
    )
    yim_dataset.vis.plot_individual_instances(
        instances=instances, class_labels=class_labels.argmax(dim=1), save=False, show=True
    )
    yim_dataset.vis.plot_instances_bb_classes(
        instances=instances,
        bounding_boxes=yim_dataset.data.bounding_box_xcycwh_to_x0y0x1y1(bounding_boxes),
        class_labels=class_labels.argmax(dim=1),
        save=False,
        show=True,
        show_class_label=True,
    )


if __name__ == "__main__":
    main()
