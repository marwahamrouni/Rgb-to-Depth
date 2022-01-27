import numpy as np # linear algebra
import torch
from torch import nn
import gc
import random
from canny_edge_detection import CannyFilter
import load_data
import timm
from torch.cuda.amp import GradScaler, autocast 
from bilinear_sampler import apply_disparity
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
real_dim = 2 # The disparity map dimension(Left and right disparities)
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
critic = Critic(input_dim).to(device) # Defining the Critic architecture
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=0.0001) # The critic's optimizer
critic = critic.apply(weights_init) # Randomly initializing the critic weights
train_loader = load_data.getTrainingData(batch_size) # loading training data

scaler1 = GradScaler() # Critic Gradient scaler for scaling the gradient during training with mixed precision (tensors of 16bits and 32bits)
scaler2 = GradScaler() # Genrator Gradient scaler

def train(save_model=False, gen_opt= gen_opt, critic_opt =critic_opt, epoch=0 ):
    mean_generator_loss = 0
    mean_critic_loss = 0
    global cur_step
    cos = nn.CosineSimilarity(dim=1, eps=0)
    for i, sample_batched in enumerate(train_loader):

        left_image, right_image = sample_batched['image'].to(dtype=torch.float16).to(device), sample_batched['depth'].to(dtype=torch.float16).to(device)

        ##Update critic## 
        for param in critic.parameters():#Use 'for param in model.parameters(): param.grad = None' speeds up the training comparing to model_opt.zero_grad()
            param.grad = None
        with autocast():
            fake_disparities =  gen(left_image) # estimated disparities
            fake_left_disp, fake_right_disp = fake_disparities[:,0,:,:].unsqueeze(1), fake_disparities[:,1,:,:].unsqueeze(1) 
            fake_right_image = apply_disparity(left_image, fake_right_disp, device=device) # Reconstructed right view
            fake_left_image = apply_disparity(right_image, -fake_left_disp, device=device) # Reconstructed left view
            mean_iteration_critic_loss = 0
            for j in range(crit_repeats):
                fake_image_pred = critic(fake_right_image.detach())
                real_image_pred = critic(right_image)
                epsilon = torch.rand(len(right_image), 1, 1, 1, device=device, requires_grad=True)
                grediant = get_grediant(critic, right_image, fake_right_image, epsilon)
                gp = grediant_penality(grediant)
                critic_loss = get_critic_loss(fake_image_pred, real_image_pred, gp, c_lambda)
                print('critic_loss:', critic_loss, 'current step:', cur_step)
                #print('fake_depths_pred:', fake_depths_pred)
                if not torch.isfinite(critic_loss):
                    print('current step:', cur_step, 'epoch:', epoch)
                    print('Scale:',scaler1.get_scale())
                    exit()
                mean_iteration_critic_loss = mean_iteration_critic_loss + critic_loss.item()/ crit_repeats

        scaler1.scale(critic_loss).backward(retain_graph=True)
        scaler1.step(critic_opt)
        scaler1.update()

        del fake_image_pred
        del real_image_pred
        gc.collect()

        ##Update generator##
        for param in gen.parameters():
            param.grad = None
        gen_loss = 0
        with autocast():
            
            fake_image_pred = critic(fake_right_image) # The score of the reconstructed right view

            #Adversarial loss
            adv = get_gen_loss(fake_image_pred)

            #L-R Consistency
            right_left_disp = apply_disparity(fake_right_disp, fake_left_disp, device=device)
            left_right_disp = apply_disparity(fake_left_disp, -fake_right_disp, device=device)
            lr_left_loss = torch.mean(torch.abs(right_left_disp - fake_left_disp))
            lr_right_loss = torch.mean(torch.abs(left_right_disp - fake_right_disp))
            lr_loss = lr_left_loss+lr_right_loss
            
            #RMSE
            left_image=torch.where(left_image <=0., torch.tensor(0.0001).to(dtype=torch.float16).to(device), left_image)
            fake_left_image=torch.where(fake_left_image <=0., torch.tensor(0.0001).to(dtype=torch.float32).to(device), fake_left_image)
            rmse_loss_left = torch.sqrt(torch.mean(torch.pow(torch.log(fake_left_image) - torch.log(left_image),2)))

            right_image=torch.where(right_image <=0., torch.tensor(0.0001).to(dtype=torch.float16).to(device), right_image)
            fake_right_image=torch.where(fake_right_image <=0., torch.tensor(0.0001).to(dtype=torch.float32).to(device), fake_right_image)
            rmse_loss_right = torch.sqrt(torch.mean(torch.pow(torch.log(fake_right_image) - torch.log(right_image),2)))


            #Grad
            lf_grad_x_fake, lf_grad_y_fake = canny_filter(fake_left_image)
            lf_grad_x_real, lf_grad_y_real = canny_filter(left_image)
            lf_gradient_loss = torch.mean(torch.pow(lf_grad_x_real - lf_grad_x_fake,2) + torch.pow(lf_grad_y_real - lf_grad_y_fake, 2))
            
            r_grad_x_fake, r_grad_y_fake = canny_filter(fake_right_image)
            r_grad_x_real, r_grad_y_real = canny_filter(right_image)
            r_gradient_loss = torch.mean(torch.pow(r_grad_x_real - r_grad_x_fake,2) + torch.pow(r_grad_y_real - r_grad_y_fake, 2))


            #Normal
            lf_ones = torch.ones(left_image.size(0), 1, left_image.size(2),left_image.size(3)).float().to(device)
            lf_real_normal = torch.cat((-lf_grad_x_real, -lf_grad_y_real, lf_ones), 1)
            lf_fake_normal = torch.cat((-lf_grad_x_fake, -lf_grad_y_fake, lf_ones), 1)
            lf_normal_loss = torch.abs(1 - cos(lf_fake_normal, lf_real_normal)).mean()

            r_ones = torch.ones(left_image.size(0), 1, left_image.size(2),left_image.size(3)).float().to(device)
            r_real_normal = torch.cat((-r_grad_x_real, -r_grad_y_real, r_ones), 1)
            r_fake_normal = torch.cat((-r_grad_x_fake, -r_grad_y_fake, r_ones), 1)
            r_normal_loss = torch.abs(1 - cos(r_fake_normal, r_real_normal)).mean()

            #Total loss
            gen_loss =   adv +  lf_gradient_loss + r_gradient_loss +   rmse_loss_left +  rmse_loss_right +  lf_normal_loss +  r_normal_loss + lr_loss

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
                  }, f"rgb2depth_{epoch}.pth")  
    del left_image
    del right_image
       
    gc.collect()
    torch.cuda.empty_cache()

for epoch in range(n_epochs):
    adjust_learning_rate(gen_opt, epoch)
    adjust_learning_rate(critic_opt, epoch)
    train(True, gen_opt, critic_opt,epoch)