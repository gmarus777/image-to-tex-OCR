import albumentations as alb
from albumentations.pytorch import ToTensorV2



train_transform =alb.Compose(
                [   alb.augmentations.geometric.resize.Resize(height=64, width= 512, p=1),
                    alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[255, 255, 255], p=1),
                    alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=255, p=0.5),
                    alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5),
                    alb.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    alb.GaussianBlur(blur_limit=(1, 1), p=0.5),
                    alb.RandomBrightnessContrast(.5, (-.5, .5), True, p=0.3),
                    alb.ImageCompression(95, p=.3),

                    ToTensorV2(),
                ]
)





test_transform = alb.Compose(
    [   alb.augmentations.geometric.resize.Resize(height=64, width= 512, p=1),
        alb.ToGray(always_apply=True),
        # alb.Sharpen(),
        ToTensorV2(),
    ]
)