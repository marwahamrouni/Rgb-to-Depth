import torch
from torch import nn




def get_gen_loss(fake_pred_critic):
    
    gen_loss = - torch.mean(fake_pred_critic)
    return gen_loss

def get_critic_loss(score_fake_depths, score_real_depth, gp, c_lambda):
    critic_loss = torch.mean(score_fake_depths) - torch.mean(score_real_depth) + gp * c_lambda
    return critic_loss    

# Compute the grediant of the critic model
def get_grediant(critic, real_depth, fake_depths, condition, epsilon):
    mixed_image = real_depth * epsilon + fake_depths * (1-epsilon)
    mixed_image_score = critic(mixed_image.to(dtype=torch.float16), condition)
    grediant = torch.autograd.grad(
                        inputs=mixed_image, 
                        outputs=mixed_image_score,
                        grad_outputs=torch.ones_like(mixed_image_score), 
                        create_graph=True,
                        retain_graph=True)[0]
    return grediant

# Compute the grediant penality
def grediant_penality(grediant):
    
    grediant = grediant.view(len(grediant),-1)
    grediant_norm = grediant.norm(2, dim=1)
    penality = torch.mean((grediant_norm -1)**2)
    return penality

# adjusting the learning rate while training  
def adjust_learning_rate(optimizer, epoch, lr):
   
    lr = lr*(0.1**(epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

#randomly initializing model's weights from a normal distribution
def weights_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)  

