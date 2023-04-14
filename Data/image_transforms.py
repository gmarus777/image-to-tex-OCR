import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
from albumentations import *

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512





class Image_Transforms:
    train_transform_with_padding = alb.Compose(
        [ alb.augmentations.geometric.resize.Resize(height=128, width=1024, p=1),
            alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.4, 0), rotate_limit=1.5, border_mode=0, interpolation=3, value=[255, 255, 255], p=.6),
            alb.Affine(scale=(0.6, 1.0), rotate=(-1, 1), cval=[255, 255, 255],interpolation=3,mode=0, p=0.4), # was  cval=255
            alb.GridDistortion(distort_limit=0.12, border_mode=0, interpolation=3, value=[255, 255, 255], p=.35),
            #alb.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=.2),
            alb.GaussNoise(var_limit=(10.0, 50.0), p=.33),
            alb.GaussianBlur(blur_limit=(1, 9), p=.33),
            alb.RandomBrightnessContrast(.1, (-.3, .1), True, p=0.25),
            alb.ImageCompression(95, p=.3),
            alb.InvertImg(p=1),
            alb.ToGray(always_apply=True),
            alb.Sharpen(p=.2),
         ToTensorV2(),
         ]
    )

    test_transform_with_padding = alb.Compose(
        [alb.augmentations.geometric.resize.Resize(height=128, width=1024, p=1),
        alb.InvertImg(p=1),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(),
         ToTensorV2(),
         ]
    )

    test_transform_with_padding_TEST = alb.Compose(
        [  #alb.augmentations.geometric.resize.Resize(height=128, width=1280, p=1),
            alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.5, 0), rotate_limit=1.5, border_mode=0, interpolation=3,value=[255, 255, 255], p=1),
            #alb.Affine(scale=(0.8, 1.0), rotate=(-1, 1), cval=0, p=1),
            alb.GridDistortion(distort_limit=0.12, border_mode=0, interpolation=3, value=[255, 255, 255], p=1),
            #alb.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
            alb.GaussNoise(var_limit=(10.0, 40.0), p=1),
            alb.GaussianBlur(blur_limit=(1, 7), p=1),
            alb.RandomBrightnessContrast(.1, (-.3, .1), True, p=1),
            alb.ImageCompression(95, p=1),

            alb.InvertImg(p=1),
            alb.ToGray(always_apply=True),
            alb.Sharpen(p=1),

            ToTensorV2(),
        ]
    )











    train_transform_with_padding_old = alb.Compose(

        [   #alb.augmentations.geometric.resize.SmallestMaxSize(max_size=96, interpolation= cv2.INTER_LINEAR ,always_apply=True) ,

            #alb.augmentations.geometric.resize.LongestMaxSize(max_size=1920, interpolation=cv2.INTER_LINEAR, always_apply=True),
            # alb.PadIfNeeded(always_apply=True, min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0),
            # alb.augmentations.crops.transforms.CenterCrop(384, 640, always_apply=True, p=1.0),
            # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),
            # alb.PadIfNeeded(always_apply=True, min_height=128, min_width=1920, border_mode=cv2.BORDER_CONSTANT, position= alb.PadIfNeeded.PositionType.TOP_LEFT, value=0),

            alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.8),
            alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=255, p=0.5),
            alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5),
            alb.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            alb.GaussianBlur(blur_limit=(1, 1), p=0.5),
            alb.RandomBrightnessContrast(.5, (-.5, .5), True, p=0.3),
            alb.ImageCompression(95, p=.3),



            #alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.2),
            #alb.Affine(scale=(.5, 1), rotate=(-.5,.5 ), cval=255, p=.35,),# keep_ratio=True
            # alb.InvertImg(p=.15),
            #alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.15),
            #alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            #alb.Affine(scale=(0.6, 1.0), rotate=(-.5, .5), cval=255, p=0.2),
            #alb.GaussNoise(10, p=0.2),
            # alb.GaussianBlur(blur_limit=(1, 1), p=0.2),
            #alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
            # alb.ImageCompression(95, p=.3),
            alb.ToGray(always_apply=True),
            # Mean:  tensor([71.5338])
            # Std: tensor([101.7121])
            # [0.485,0.456,0.406], [0.229,0.224,0.225]
            #alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            #alb.Normalize(),
            #alb.Sharpen(p=.05),
            # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),

            ToTensorV2(),
        ]
    )



    test_transform_with_padding_OLD = alb.Compose(

        [
            # alb.augmentations.geometric.resize.LongestMaxSize (max_size=450, interpolation= cv2.INTER_CUBIC, always_apply=True, p=1),
            #alb.augmentations.geometric.resize.SmallestMaxSize(max_size=96, interpolation= cv2.INTER_LINEAR ,always_apply=True, p=1),
            # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
            # alb.PadIfNeeded(always_apply=True, min_height=128, min_width=1920, border_mode=cv2.BORDER_CONSTANT, position=alb.PadIfNeeded.PositionType.TOP_LEFT, value=0),
            # alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

            # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
            #alb.ImageCompression(50, p=1),
            alb.ToGray(always_apply=True),
            #alb.Normalize(),
            #alb.Sharpen(always_apply=True  ),
            ToTensorV2(),
        ]
    )

    test_transform_with_padding_SMALL = alb.Compose(

        [
            # alb.augmentations.geometric.resize.LongestMaxSize (max_size=450, interpolation= cv2.INTER_CUBIC, always_apply=True, p=1),
            alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_LINEAR,always_apply=True, p=1),
            # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
            # alb.PadIfNeeded(always_apply=True, min_height=128, min_width=1920, border_mode=cv2.BORDER_CONSTANT, position=alb.PadIfNeeded.PositionType.TOP_LEFT, value=0),
            # alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

            # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
            # alb.ImageCompression(95, p=1),
            alb.ToGray(always_apply=True),
            # alb.Normalize(),
            #alb.Sharpen(always_apply=True),
            ToTensorV2(),
        ]
    )





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



    train_transform_with_padding_small = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=450, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

         alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[0, 0, 0], p=.3),
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         # alb.InvertImg(p=.15),
         alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[0, 0, 0], p=.15),
         #alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
         alb.GaussNoise(10, p=0.2),
         # alb.GaussianBlur(blur_limit=(1, 1), p=0.2),
         alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
         #alb.ImageCompression(95, p=.3),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(always_apply=True),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),

         ToTensorV2(),
         ]
    )

    train_transform_with_padding_xs = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=300, interpolation=cv2.INTER_CUBIC,
                                                           always_apply=True, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

         alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3, value=[0, 0, 0], p=.3),
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         # alb.InvertImg(p=.15),
         alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[0, 0, 0], p=.15),
         # alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
         alb.GaussNoise(10, p=0.2),
         # alb.GaussianBlur(blur_limit=(1, 1), p=0.2),
         alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
         # alb.ImageCompression(95, p=.3),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(always_apply=True),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),

         ToTensorV2(),
         ]
    )




        # keep the similar ration for all images  and then pad
        # rescale all photos by to a common ratio aspect
    # if size is small, then upscale to around
    # after padd to certain size
    test_transform_with_padding_old = alb.Compose(

        [ #alb.augmentations.geometric.resize.LongestMaxSize (max_size=450, interpolation= cv2.INTER_CUBIC, always_apply=True, p=1),
          #alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
            #alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
            #alb.PadIfNeeded(always_apply=True, min_height=128, min_width=1920, border_mode=cv2.BORDER_CONSTANT, position=alb.PadIfNeeded.PositionType.TOP_LEFT, value=0),
            #alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

          #alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
          #alb.ImageCompression(95, p=.3),
         alb.ToGray(always_apply=True),
         alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
         #alb.Sharpen(always_apply=True  ),
         ToTensorV2(),
         ]
    )

    test_transform_with_padding_small = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=350, interpolation=cv2.INTER_CUBIC,always_apply=True, p=1),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
         # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

         #alb.transforms.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.5)
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         alb.ToGray(always_apply=True),
         alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
          #alb.Sharpen(),
         ToTensorV2(),
         ]
    )


    test_transform_with_padding_medium = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=400, interpolation=cv2.INTER_CUBIC,
                                                           always_apply=True, p=1),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
         # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

         # alb.transforms.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.5)
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(),
         ToTensorV2(),
         ]
    )
    # 1 width is 0.6
    test_transform_with_padding_xl = alb.Compose(

        [alb.augmentations.geometric.resize.LongestMaxSize(max_size=400, interpolation=cv2.INTER_AREA,
                                                           always_apply=True, p=1),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
         # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

         # alb.transforms.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.5)
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(),
         ToTensorV2(),
         ]
    )

    test_transform_with_padding_xs = alb.Compose(

        [#alb.augmentations.geometric.resize.LongestMaxSize(max_size=300, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),
         # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
         # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
         alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
         alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

         # alb.transforms.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.5)
         # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
         alb.ToGray(always_apply=True),
         #alb.Sharpen(),
         ToTensorV2(),
         ]
    )

    test_transform_with_padding_xss = alb.Compose(

        [
            alb.augmentations.geometric.resize.LongestMaxSize(max_size=300, interpolation=cv2.INTER_CUBIC, always_apply=True, p=1),
            # alb.augmentations.geometric.resize.SmallestMaxSize(max_size=64, interpolation= cv2.INTER_CUBIC ,always_apply=False, p=1),
            # alb.augmentations.geometric.resize.Resize(interpolation= cv2.INTER_CUBIC, height=30, width= 217, p=1),
            alb.PadIfNeeded(always_apply=True, min_height=600, min_width=600, border_mode=cv2.BORDER_CONSTANT, value=0),
            alb.augmentations.crops.transforms.CenterCrop(350, 600, always_apply=True, p=1.0),

            # alb.transforms.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=0.5)
            # alb.Affine(scale=(0.6, 1.0), rotate=(-2, 2), cval=0, p=0.5),
            alb.ToGray(always_apply=True),
            # alb.Sharpen(),
            ToTensorV2(),
        ]
    )