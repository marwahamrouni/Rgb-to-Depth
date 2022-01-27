import numpy as np
from numpy.lib.type_check import real # linear algebra
import torch
from torch import nn
from torchvision import transforms
import random
import load_data
import torch.nn.functional as F
from PIL import Image
import os
from PIL import Image
import timm
from network import *
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

##Post-processing##
def predict_tta(model, image):
    pred = model(image)
    pred = np.clip(pred.cpu().detach().numpy(), min_depth, max_depth)
    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)
    pred_lr = model(image)
    pred_lr = np.clip(pred_lr.cpu().detach().numpy()[..., ::-1], min_depth, max_depth)
    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), (480, 640), mode='bilinear', align_corners=True)# upsample the generated depth to the size of the ground truth depth 
    return torch.Tensor(final)


device = torch.device('cuda:1')
input_dim = 3 # RGB image dimension
real_dim = 1  # Depth map dimension


basemodel_name = 'tf_efficientnet_b5_ap'

print('Loading base model ()...'.format(basemodel_name), end='')
encoder = timm.create_model(basemodel_name, features_only=True, pretrained=True)
print('Done.')

gen = UNet(real_dim, encoder).to(device)


batch_size =1
max_depth = 1 
min_depth = 0.0001   
metrics = RunningAverageDict()
test_loader, dataset  = load_data.getTestingData(batch_size)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


if dataset=='kitti':
    eigen_crop = 0
    garg_crop = 1
    factor = 20070
elif dataset == 'nyu2':
    eigen_crop = 1
    garg_crop = 0
    factor = 10000   


model = torch.load(f"img2depth32_2.pth", device)# loading the pre-trained model
gen.load_state_dict(model['gen'])
for param in gen.parameters():
    param.grad = None
gen.eval()
    
metrics = RunningAverageDict()
   
    
for j, sampled_batch in enumerate(test_loader):
        
    condition, real_depth = sampled_batch['image'].to(device), sampled_batch['depth'].to(device)
    ##when training on KITTI dataset the following lines should be uncommented to resize the RGB images and depth maps
    #real_depth = nn.functional.interpolate(real_depth, (condition.shape[2], condition.shape[3]), mode='bilinear', align_corners=True)
    #condition = transforms.functional.resize(condition,(160, 512), 2)

    fake_depth = predict_tta(gen, normalize(condition)) * 10
    #fake_depth = nn.functional.interpolate(fake_depth, (real_depth.shape[2], real_depth.shape[3]), mode='bilinear', align_corners=True)      
    fake_depth = fake_depth.squeeze().cpu().numpy()
    fake_depth[np.isinf(fake_depth)] = max_depth * 10
    fake_depth[np.isnan(fake_depth)] = min_depth * 10
    real_depth = real_depth.squeeze().cpu().numpy()
    dir_ = f"Results/"
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    path_fake = os.path.join(dir_,f"fakedepth_{j}.png") 
    real = os.path.join(dir_,f"realdepth_{j}.png")
    path_condition = os.path.join(dir_,f"condition_{j}.png") 
    Image.fromarray(real_depth).save(real)
    Image.fromarray(fake_depth).save(path_fake) 
    transforms.ToPILImage()(condition.squeeze()).save(path_condition)

    ##Evaluating on cropped images##
    valid_mask = np.logical_and(real_depth > min_depth * 80, real_depth < max_depth * 80)
       
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


