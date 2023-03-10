import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2


# alb.augmentations.geometric.resize.Resize(height=64, width= 512, p=1),
# albumentations.augmentations.geometric.resize.LongestMaxSize
# albumentations.augmentations.geometric.resize.SmallestMaxSize
train_transform_original=alb.Compose(
                    [
                        alb.augmentations.geometric.resize.LongestMaxSize (max_size=1024, interpolation=1, p=1),
                        alb.Affine(scale=(0.6, 1.0), rotate=(-1, 1), cval=255, p=0.5),
                        alb.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        alb.GaussianBlur(blur_limit=(1, 1), p=0.5),
                        ToTensorV2(),

                    ]
    )




#  alb.PadIfNeeded(min_height=128, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=255),

train_transform =alb.Compose(
                    [
                        alb.augmentations.geometric.resize.LongestMaxSize(max_size=512, interpolation=1, p=1),
                        alb.augmentations.geometric.resize.SmallestMaxSize(max_size=128, interpolation=1, p=1),
                        alb.augmentations.geometric.resize.Resize(128, 512, interpolation=1, p=1),
                        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[255, 255, 255], p=1),
                        alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=255, p=0.5),
                        alb.GridDistortion(distort_limit=0.2, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5),
                        alb.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        alb.GaussianBlur(blur_limit=(1, 1), p=0.5),
                        alb.RandomBrightnessContrast(.5, (-.5, .5), True, p=0.3),
                        alb.ImageCompression(95, p=.3),

                        ToTensorV2(),
                    ]
    )




#  alb.PadIfNeeded(min_height=256, min_width=1024, border_mode=cv2.BORDER_CONSTANT, value=255),
test_transform = alb.Compose(
    [
        alb.augmentations.geometric.resize.LongestMaxSize (max_size=512, interpolation=1, p=1),
        alb.augmentations.geometric.resize.SmallestMaxSize (max_size=128, interpolation=1, p=1),
        alb.augmentations.geometric.resize.Resize (128, 512, interpolation=1,  p=1),
        # alb.Sharpen(),
        ToTensorV2(),
    ]
)










def yield_image_transforms_train():
    return train_transform

def yield_image_transforms_test():
    return test_transform



