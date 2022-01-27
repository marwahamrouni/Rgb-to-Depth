import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from data_transform import *
from transform_stereo_kitti import *
class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=64):
 
    ##When we train on stereo images from the KITTI dataset##
    transformed_training = depthDataset(csv_file='data/kitti_stereo_eigen_train.csv',
                                         transform=image_transforms_kitti())

    ##When we train on the NYU-Depth-v2 dataset##
    #transformed_training = depthDataset(csv_file='data/nyu2_train_1.csv',
    #                                    transform=transforms.Compose([
    #                                        Scale(240),
    #                                        RandomHorizontalFlip(),
    #                                        RandomRotate(5),
    #                                        ToTensor(),
    #                                        ColorJitter(
    #                                            brightness=0.4,
    #                                           contrast=0.4,
    #                                            saturation=0.4)
    #                                    ]))

    ##When we train on the KITTI dataset (supervised method)##
    #transformed_training = depthDataset(csv_file='data/kitti_train_gt.csv',
    #                                    transform=transforms.Compose([                                   
    #                                        RandomHorizontalFlip(),
    #                                        ToTensor(),
    #                                        ColorJitter(
    #                                            brightness=0.4,
    #                                           contrast=0.4,
    #                                            saturation=0.4)
    #                                    ]))
                                     
    dataloader_training = DataLoader(transformed_training, batch_size, num_workers=1,
                                     shuffle=True, pin_memory=True)

    return dataloader_training


def getTestingData(batch_size=5):

    file_name = 'data/nyu2_test_1.csv'
    dataset = file_name.split('/')[-1].split('_')[0]
    if dataset == 'nyu2':
        transformed_testing = depthDataset(csv_file=file_name,
                                       transform=transforms.Compose([
                            
                                           Scale(240, True),
                                           ToTensor(is_test=True)
                         
                                       ]))
    elif dataset == 'kitti':
        transformed_testing = depthDataset(csv_file=file_name,
                                       transform=transforms.Compose([
                                           ToTensor(is_test=True)
                         
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False,pin_memory=False)
    
    return dataloader_testing, dataset