import os

import PIL
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
from imgaug import augmenters as iaa
import dlib

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Split into train/validation/test sets so that all faces from one image
# belong to the same set


def split_image_name(val):
    return val.split('/')[-1]


def relabel(frame):
    # relabel races for training
    if 'age' in frame.columns:
        # frame.loc[frame['age'] == '0-2', 'age'] = 0
        # frame.loc[frame['age'] == '3-9', 'age'] = 1
        # frame.loc[frame['age'] == '10-19', 'age'] = 2
        # frame.loc[frame['age'] == '20-29', 'age'] = 3
        # frame.loc[frame['age'] == '30-39', 'age'] = 4
        # frame.loc[frame['age'] == '40-49', 'age'] = 5
        # frame.loc[frame['age'] == '50-59', 'age'] = 6
        # frame.loc[frame['age'] == '60-69', 'age'] = 7
        # frame.loc[frame['age'] == 'more than 70', 'age'] = 8

        frame.loc[frame['age'] == '0-2', 'age'] = 0
        frame.loc[frame['age'] == '3-9', 'age'] = 0
        frame.loc[frame['age'] == 'more than 70', 'age'] = 0
        frame.loc[frame['age'] == '10-19', 'age'] = 1
        frame.loc[frame['age'] == '20-29', 'age'] = 2
        frame.loc[frame['age'] == '30-39', 'age'] = 3
        frame.loc[frame['age'] == '40-49', 'age'] = 4
        frame.loc[frame['age'] == '50-59', 'age'] = 5
        frame.loc[frame['age'] == '60-69', 'age'] = 6

    # gender label
    if 'gender' in frame.columns:
        frame.loc[frame['gender'] == 'Male', 'gender'] = 0
        frame.loc[frame['gender'] == 'Female', 'gender'] = 1

    if 'age' in frame.columns and 'gender' in frame.columns:
        n_races = max(frame['age']) + 1
        frame['ageAndgender'] = frame['age'] + frame['gender'] * n_races

    return frame


def add_imbalance(frame):
    if 'race' not in frame.columns:
        print("No race label found! Imbalance not added.")
        return frame

    nrow = frame.shape[0]

    np.random.seed(42)
    choice = np.random.rand(nrow)
    # Drop 0.8 of non-White images.
    frame = frame[((np.array(frame['race'] == 0))) | (
        (np.array(frame['race'] != 0)) & (choice >= 0.8))]
    frame = frame.reset_index(drop=True)
    return frame


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
# image.
def sometimes(aug): return iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every
# image.


class ImgAugTransform:

    def __init__(self):
        self.aug = iaa.Sequential(
            [
                #
                # Apply the following augmenters to most images.
                #
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images

                # crop some of the images by 0-10% of their height/width
                sometimes(iaa.Crop(percent=(0, 0.05))),

                # Apply affine transformations to some of the images
                # - scale to 80-120% of image height/width (each axis independently)
                # - translate by -20 to +20 relative to height/width (per axis)
                # - rotate by -45 to +45 degrees
                # - shear by -16 to +16 degrees
                # - order: use nearest neighbour or bilinear interpolation (fast)
                # - mode: use any available mode to fill newly created pixels
                #         see API or scikit-image for which modes are available
                # - cval: if the mode is constant, then use a random brightness
                #         for the newly created pixels (e.g. sometimes black,
                #         sometimes white)
                iaa.Affine(
                    scale={"x": (1, 1.1), "y": (1, 1.1)},  # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -10 to +10 percent (per axis)
                    rotate=(-15, 15),  # rotate by -15 to +15 degrees
                    shear=(-8, 8),  # shear by -8 to +8 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=['edge']  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                ),

                #
                # Execute 0 to 5 of the following (less important) augmenters per
                # image. Don't execute all of them, as that would often be way too
                # strong.
                #
                iaa.SomeOf((0, 5),
                           [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 0.1),
                            n_segments=(50, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 0.3), strength=(0, 2)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.3)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.3), direction=(0.0, 1.0)
                        ),
                    ])),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.02), per_channel=0.5),
                        # iaa.CoarseDropout(
                        #    (0.03, 0.15), size_percent=(0.02, 0.05),
                        #    per_channel=0.2
                        # ),
                    ]),

                    # Invert each image's chanell with 5% probability.
                    # This sets each pixel value v to 255-v.
                    # iaa.Invert(0.05, per_channel=True), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-15, 15), per_channel=0.5),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.75, 1.25), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.ContrastNormalization((0.75, 1.75), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    # sometimes(
                    #    iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=0.25)
                    # ),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.005, 0.01)))
                ],
                    # do all of the above augmentations in random order
                    random_order=True
                )
            ],
            # do all of the above augmentations in random order
            random_order=True
        )

    def __call__(self, img):

        img = np.array(img)
        return self.aug.augment_image(img)


class FaceDataset(Dataset):

    def __init__(
        self,
        data_frame,
        transform=None,
        col_used=None):

        self.data_frame = data_frame
        self.transform = transform
        self.col_used = col_used
    def __len__(self):

        return len(self.data_frame)

    def __getitem__(self, idx):
        # idx is index from dataset
        # This is a mapping from your data_frame to the output of the mode
        img_name = self.data_frame.loc[idx, 'face_name_align']
        label = self.data_frame.loc[idx, self.col_used]

        # read image as ndarray, H*W*C
        image = dlib.load_rgb_image(img_name)

        if self.transform:
            image = self.transform(image)

        # transform label to torch tensor
        # This sets the order of the label
        return image, label


def make_frame(
        csv,
        face_dir,
        imbalance=False):
    frame = pd.read_csv(csv)
    frame.head()

    frame = relabel(frame)

    if imbalance:
        frame = add_imbalance(frame)

    # Change face_name_align if the images are now stored in a different dir
    # Also make sure all faces are found and can be
    if face_dir:
        initial_rows = frame.shape[0]
        faces = set(os.listdir(face_dir))
        faces_found = 0
        new_face_name = []
        face_found_mask = []
        for i in range(frame.shape[0]):
            face_name_align = split_image_name(frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                new_path = os.path.join(face_dir, face_name_align)
                try:
                    faces_found += 1
                    new_face_name.append(new_path)
                except BaseException:
                    continue
        frame = frame[face_found_mask].reset_index(drop=True)
        frame['face_name_align'] = new_face_name
        print(
            "{} out of {} faces are found in new dir!".format(
                faces_found,
                initial_rows))

    image_name_frame = frame['image_name'].apply(split_image_name)
    image_names = image_name_frame.unique()
    np.random.seed(42)
    image_names = np.random.permutation(image_names)
    train_data = frame[image_name_frame.isin(image_names)].reset_index(drop=True)
    return train_data


def make_datasets(data_frame,
        input_size = 224,   
        give_dataloader=True,
        batch_size=64,
        col_used=None):
    
    transform_train_data = transforms.Compose([
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed_dataset = FaceDataset(data_frame=data_frame, transform=transform_train_data, col_used=col_used)
    
    if give_dataloader:
        dataloader = DataLoader(
        transformed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8)
        return dataloader
    else:
        
        return transformed_dataset
