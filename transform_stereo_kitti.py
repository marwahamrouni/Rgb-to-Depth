##Data augmentation to stereo images from the KITTI dataset##
from torchvision import transforms
import numpy as np
import torch

def image_transforms_kitti(mode='train', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None):
    if mode == 'train':
        data_transform = transforms.Compose([
            
            RandomFlip(do_augmentation),
            ToTensor2(train=True),
            AugmentImagePair(augment_parameters, do_augmentation)
        ])
    return data_transform


class ToTensor2(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            left_image = sample['image']
            right_image = sample['depth']
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {'image': new_left_image,
                      'depth': new_right_image}
        else:
            left_image = sample['image']
            sample = self.transform(left_image)
        return sample

class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample['image']
        right_image = sample['depth']
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(right_image)
                fliped_right = self.transform(left_image)
                sample = {'image': fliped_left, 'depth': fliped_right}
        else:
            sample = {'image': left_image, 'depth': right_image}
        return sample

class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        left_image = sample['image']
        right_image = sample['depth']
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # randomly shift gamma
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = left_image ** random_gamma
                right_image_aug = right_image ** random_gamma

                # randomly shift brightness
                random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                left_image_aug = left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # randomly shift color
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_image_aug[i, :, :] *= random_colors[i]
                    right_image_aug[i, :, :] *= random_colors[i]

                # saturate
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = {'image': left_image_aug, 'depth': right_image_aug}

        else:
            sample = {'image': left_image, 'depth': right_image}
        return sample