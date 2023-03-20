import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512




train_transform =alb.Compose(
                    [   alb.augmentations.geometric.resize.Resize(height=IMAGE_HEIGHT, width= IMAGE_WIDTH, p=1),
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
    [   alb.augmentations.geometric.resize.Resize(height=IMAGE_HEIGHT, width= IMAGE_WIDTH, p=1),
        alb.ToGray(always_apply=True),
        # alb.Sharpen(),
        ToTensorV2(),
    ]
)

class Image_Transforms:

    train_transform =alb.Compose(
                    [   alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_LINEAR,height=IMAGE_HEIGHT, width= IMAGE_WIDTH, p=1),
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

    train_transform_with_padding = alb.Compose(

                    [ alb.augmentations.geometric.resize.LongestMaxSize(max_size=420, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),
                      alb.PadIfNeeded(always_apply=True, min_height=420, min_width=420, border_mode=cv2.BORDER_CONSTANT, value=0),
                      alb.augmentations.crops.transforms.CenterCrop(250, 420, always_apply=True, p=1.0),

                        alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[0, 0, 0], p=1),
                        #alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
                        # alb.InvertImg(p=.15),
                        alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[0, 0, 0], p=.15),
                        alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                        alb.GaussNoise(10, p=0.2),
                        #alb.GaussianBlur(blur_limit=(1, 1), p=0.2),
                        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                        alb.ImageCompression(95, p=.3),
                        alb.ToGray(always_apply=True),
                        alb.Sharpen(always_apply=True),
                        # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),



                        ToTensorV2(),
                    ]
    )

    train_transform_with_padding_small = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=200, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=420, min_width=420, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(250, 420, always_apply=True, p=1.0),

         alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3,
                              value=[0, 0, 0], p=1),
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         # alb.InvertImg(p=.15),
         alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[0, 0, 0], p=.15),
         alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
         alb.GaussNoise(10, p=0.2),
         # alb.GaussianBlur(blur_limit=(1, 1), p=0.2),
         alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
         alb.ImageCompression(95, p=.3),
         alb.ToGray(always_apply=True),
         alb.Sharpen(always_apply=True),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),

         ToTensorV2(),
         ]
    )





    test_transform = alb.Compose(
        [   alb.augmentations.geometric.resize.Resize(height=IMAGE_HEIGHT, width= IMAGE_WIDTH, p=1),
            alb.ToGray(always_apply=True),
            # alb.Sharpen(),
            ToTensorV2(),
        ]
    )

    test_transform_with_padding = alb.Compose(

        [ alb.augmentations.geometric.resize.LongestMaxSize (max_size=420, interpolation= cv2.INTER_AREA, always_apply=True, p=1),
          #alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
            #alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
            alb.PadIfNeeded(always_apply=True, min_height=420, min_width=420, border_mode=cv2.BORDER_CONSTANT, value=1),
          alb.augmentations.crops.transforms.CenterCrop(250, 420, always_apply=True, p=1.0),

          #alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
          #alb.ImageCompression(95, p=.3),
         alb.ToGray(always_apply=True),
         alb.Sharpen(always_apply=True  ),
         ToTensorV2(),
         ]
    )

    test_transform_with_padding_small = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=200, interpolation=cv2.INTER_CUBIC,always_apply=True, p=1),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
         # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=420, min_width=420, border_mode=cv2.BORDER_CONSTANT, value=1),
         alb.augmentations.crops.transforms.CenterCrop(250, 420, always_apply=True, p=1.0),

         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         alb.ToGray(always_apply=True),
          alb.Sharpen(),
         ToTensorV2(),
         ]
    )