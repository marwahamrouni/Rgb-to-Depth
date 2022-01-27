import numpy as np # linear algebra
import torch
from torch import nn
from torchvision import transforms
import gc
import random
from canny_edge_detection import CannyFilter
import load_data
import timm
from torch.cuda.amp import GradScaler, autocast 
from network import *
from utils import *
random.seed(1)
torch.backends.cudnn.benchmark = True #Benchmark mode 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


batch_size = 16
n_epochs = 20
device = 'cuda:0'
canny_filter = CannyFilter(device = device)
c_lambda = 10
input_dim = 3 # The RGB image dimension
real_dim = 1 # The depth map dimension
display_step = 100
lr = 0.0001 
crit_repeats = 3
cur_step = 0 # current training step

# Loading the pre-trained model
basemodel_name = 'tf_efficientnet_b5_ap'
print('Loading base model ()...'.format(basemodel_name), end='')
encoder = timm.create_model(basemodel_name, features_only=True, pretrained=True) 
print('Done.')

gen = UNet(real_dim, encoder).to(device) # The generator model
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, weight_decay=0.0001)# Defining the generator's optimizer
critic = Critic(input_dim+real_dim).to(device) # Defining the Critic architecture
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=0.0001) # The critic's optimizer
critic = critic.apply(weights_init) # Randomly initializing the critic weights
train_loader = load_data.getTrainingData(batch_size) # loading training data

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

scaler1 = GradScaler() # Critic Gradient scaler for scaling the gradient during training with mixed precision (tensors of 16bits and 32bits)
scaler2 = GradScaler() # Genrator Gradient scaler

def train(save_model=False, gen_opt= gen_opt, critic_opt =critic_opt, epoch=0 ):
    mean_generator_loss = 0
    mean_critic_loss = 0
    global cur_step
    cos = nn.CosineSimilarity(dim=1, eps=0)
    for i, sample_batched in enumerate(train_loader):
        condition, real_depth = sample_batched['image'].to(dtype=torch.float16).to(device), sample_batched['depth'].to(dtype=torch.float16).to(device)
        condition_norm = normalize(condition) # Normalize the input image before feeding it to the pre-trained model

        ##Update critic## 
        critic_opt.zero_grad()
        with autocast(): # For mixed precision training
            fake_depths = gen(condition_norm)
            mean_iteration_critic_loss = 0
            for j in range(crit_repeats):
                fake_depths_pred = critic(fake_depths.detach(), condition) # Critic evaluation on estimated depth maps 
                real_depth_pred = critic(real_depth, condition) # Critic evaluation on ground truth depth maps
                epsilon = torch.rand(len(real_depth), 1, 1, 1, device=device, requires_grad=True)
                grediant = get_grediant(critic, real_depth, fake_depths, condition, epsilon)
                gp = grediant_penality(grediant)
                critic_loss = get_critic_loss(fake_depths_pred, real_depth_pred, gp, c_lambda)
                mean_iteration_critic_loss = mean_iteration_critic_loss + critic_loss.item()/ crit_repeats
        scaler1.scale(critic_loss).backward(retain_graph=True) # Backpropagate the grediant
        scaler1.step(critic_opt) # Implicitly unscale the scaled grediants
        scaler1.update() # Update the Critic parameters

        del fake_depths_pred
        del real_depth_pred
        gc.collect()

        ##Update generator##
        gen_opt.zero_grad() 
        gen_loss =0
        with autocast():
            fake_depths_pred = critic(fake_depths, condition)
            # The adversarial loss
            adv = get_gen_loss(fake_depths_pred)

            # RMSE loss in the log space
            rmse_log_loss = torch.sqrt(torch.mean(torch.pow((torch.log(fake_depths+ 0.000001) - torch.log(real_depth+ 0.000001)),2)))

            # Gradient loss
            grad_x_fake, grad_y_fake = canny_filter(fake_depths)
            grad_x_real, grad_y_real = canny_filter(real_depth)
            gradient_loss = torch.mean(torch.pow(grad_x_real - grad_x_fake,2) + torch.pow(grad_y_real - grad_y_fake, 2))  
            
            # Normal loss
            ones = torch.ones(real_depth.size(0), 1, real_depth.size(2),real_depth.size(3)).float().to(device)
            real_normal = torch.cat((-grad_x_real, -grad_y_real, ones), 1)
            fake_normal = torch.cat((-grad_x_fake, -grad_y_fake, ones), 1)
            normal_loss = torch.abs(1 - cos(fake_normal, real_normal)).mean()

            # Total loss
            gen_loss =  adv + gradient_loss + normal_loss + rmse_log_loss
      
        print('gen loss:', gen_loss)
        scaler2.scale(gen_loss).backward()
        scaler2.step(gen_opt)
        scaler2.update()
       
        # Keep track of the average critic loss
        mean_critic_loss = mean_critic_loss + mean_iteration_critic_loss / display_step
        # Keep track of the average generator loss
        mean_generator_loss = mean_generator_loss + gen_loss.item() / display_step  
        if cur_step % display_step == 0:
            if cur_step > 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss} and critic loss: {mean_critic_loss}")
            else:
                print("Pretrained initial state")
                
            mean_generator_loss = 0
            mean_critic_loss = 0
        cur_step += 1

    if save_model:
        torch.save({'gen': gen.state_dict(),
                  }, f"img2depth_{epoch}.pth")  

    del condition
    del real_depth
    del fake_depths
       
    gc.collect()
    torch.cuda.empty_cache()

for epoch in range(n_epochs):
    adjust_learning_rate(gen_opt, epoch)
    adjust_learning_rate(critic_opt, epoch)
    train(True, gen_opt, critic_opt,epoch)