import numpy as np
from numpy.lib.type_check import real # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
from torchvision import transforms
from network import UNet
import random
import load_data
import torch.nn.functional as F
from PIL import Image
import os
from PIL import Image
import timm
random.seed(1)

torch.backends.cudnn.benchmark = False


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg

class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

##Evaluation metrics##
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

##Disparity Post-processing##
def predict_tta(model, image):
    disparities = model(image)
    disparities = disparities.detach().cpu().squeeze().numpy()
    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)
    pred_lr = model(image)
    pred_lr = pred_lr[:,0,...].detach().cpu().numpy()
    _,h, w= disparities.shape
    l_disp = disparities[0,...]
    r_disp = np.fliplr(pred_lr.squeeze())
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    pp_disp = r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
    return torch.Tensor(pp_disp).unsqueeze(0).unsqueeze(1)


#Read the camera calibration files
def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_focal_length_baseline(calib_dir):
    cam =2 # cam 2 is the left camera
    cam2cam = read_calib_file(calib_dir)
    P2_rect = cam2cam['P_rect_02'].reshape(3,4)
    P3_rect = cam2cam['P_rect_03'].reshape(3,4)

    # cam 2 is left of camera 0  -6cm
    # cam 3 is to the right  +54cm
    b2 = P2_rect[0,3] / -P2_rect[0,0]
    b3 = P3_rect[0,3] / -P3_rect[0,0]
    baseline = b3-b2

    if cam==2:
        focal_length = P2_rect[0,0]
    elif cam==3:
        focal_length = P3_rect[0,0]

    return focal_length, baseline


device = torch.device('cuda:1')
input_dim = 3
real_dim = 2

basemodel_name = 'tf_efficientnet_b5_ap'
print('Loading base model ()...'.format(basemodel_name), end='')
encoder = timm.create_model(basemodel_name, features_only=True, pretrained=True)
print('Done')

gen = UNet(real_dim, encoder).to(device)

batch_size =1
max_depth = 1. 
min_depth = 0.0001   
metrics = RunningAverageDict()
test_loader, dataset  = load_data.getTestingData(batch_size)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

print('dataset:', dataset)
if dataset=='kitti':
    eigen_crop = 0
    garg_crop = 1
    factor = 20070
elif dataset == 'nyu2':
    eigen_crop = 1
    garg_crop = 0
    factor = 10000
com =0   

model = torch.load(f"img2depth_{i}.pth", device) 
gen.load_state_dict(model['gen'])
for param in gen.parameters():
    param.grad = None
gen.eval()
metrics = RunningAverageDict()
    
for j, sampled_batch in enumerate(test_loader):
        
    condition, real_depth = sampled_batch['image'].to(device), sampled_batch['depth'].to(device)
    condition_width = condition.shape[3]
    real_depth = nn.functional.interpolate(real_depth, (condition.shape[2], condition.shape[3]), mode='bilinear', align_corners=True)
    condition_resized = transforms.functional.resize(condition,(160, 512), 2)
       
    if condition_width == 1242:
        file_path = '/home/data/calib_2011_09_26.txt'
    elif condition_width == 1224:
        file_path = '/home/data/calib_2011_09_28.txt'
    elif condition_width == 1238:
        file_path = '/home/data/calib_2011_09_29.txt'
    elif condition_width == 1226:
        file_path = '/home/data/calib_2011_09_30.txt'
    elif condition_width == 1241:
        file_path = '/home/data/calib_2011_10_03.txt'
       

    fake_disp = predict_tta(gen, condition_resized)  #The estimated disparity 
    fake_disp = nn.functional.interpolate(fake_disp, (condition.shape[2], condition.shape[3]), mode='bilinear', align_corners=True)
    fake_disp = fake_disp * fake_disp.shape[-1]
    focal_length, baseline = get_focal_length_baseline(file_path)

    fake_depth = ((focal_length * baseline) / fake_disp) # Convert disparity into a depth map
    fake_depth = fake_depth.squeeze().cpu().numpy().astype('uint16')
    fake_depth[np.isinf(fake_depth)] = 1
    fake_depth[np.isnan(fake_depth)] = 1
    fake_depth[fake_depth<1] = 1
    real_depth = (real_depth * 80).squeeze().cpu().numpy().astype('uint16')

    ##Save the generated depth maps along with the RGB images and the ground truth depth maps 
    dir_ = f"Results/"
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    path_fake = os.path.join(dir_,f"fakedepth_{j}.png") 
    real = os.path.join(dir_,f"realdepth_{j}.png")
    path_condition = os.path.join(dir_,f"condition_{j}.png") 
    Image.fromarray(real_depth).save(real)
    Image.fromarray(fake_depth).save(path_fake) 
    transforms.ToPILImage()(condition.squeeze()).save(path_condition)


    ##Evaluating on cropped images
    valid_mask = np.logical_and(real_depth > 0, real_depth < 80)  
    if garg_crop or eigen_crop:
        gt_height, gt_width = real_depth.shape
        eval_mask = np.zeros(valid_mask.shape)
        if garg_crop:
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        elif eigen_crop:
            if dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask[45:471, 41:601] = 1
    valid_mask = np.logical_and(valid_mask, eval_mask)          
    metrics.update(compute_errors(real_depth[valid_mask], fake_depth[valid_mask]))
    
metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
print(f"Metrics epoch: {metrics}")


