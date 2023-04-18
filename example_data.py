import kornia.augmentation
from torch.utils.data import DataLoader

import yim_dataset


DATASET_PATH: str = "/Users/christoph/Desktop/yeast_cell_in_microstructures_dataset/dataset/train"

def main() -> None:
    # Init augmentations
    transforms = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.RandomHorizontalFlip(p=1.0),
        data_keys=["input", "bbox_xyxy", "mask"],
        same_on_batch=False,
    )
    # Init dataset
    dataset = yim_dataset.data.YIMDataset(path=DATASET_PATH, augmentations=transforms)
    # Make data loader
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=2,
        batch_size=2,
        drop_last=True,
        collate_fn=yim_dataset.data.collate_function_yim_dataset,
    )
    # Loop over data loader
    for index, (images, instances, bounding_boxes, class_labels) in enumerate(data_loader):
        print(index, images.shape, len(instances), len(bounding_boxes), len(class_labels))


if __name__ == "__main__":
    main()
