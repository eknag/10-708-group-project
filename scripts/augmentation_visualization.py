from pythae.data.datasets import FolderDataset
from pythae.models.base.base_utils import AugmentationProcessor
import cv2

dataset = FolderDataset("../data/rgb_dataset")
print(dataset.filenames[3])
sample = dataset[3]["data"].reshape([1, 3, 64, 64])

augmentation_processor = AugmentationProcessor()
aug_names = augmentation_processor.augmentation_names

for aug in aug_names:
    aug_func = augmentation_processor.get_augmentation(aug)
    aug_img = aug_func(sample).reshape([3, 64, 64]).permute(1, 2, 0).cpu().numpy() * 255
    cv2.imwrite(f"visualization/{aug}_result.png", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
    
aug_img = sample.reshape([3, 64, 64]).permute(1, 2, 0).cpu().numpy() * 255
cv2.imwrite(f"visualization/original_result.png", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))