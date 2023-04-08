import albumentations as alb
from albumentations.pytorch import ToTensorV2

class Image_Transforms:


    test_transform_with_padding = alb.Compose(
        [#alb.augmentations.geometric.resize.Resize(height=128, width=1280, p=1),
        alb.InvertImg(p=1),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(),
         ToTensorV2(),
         ]
    )





