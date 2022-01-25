import torch
from torch import nn
import numpy as np



def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D
    
def get_sobel_kernel(k=3):
    # get range
        range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D*2

class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device='cuda:0'):
        super(CannyFilter, self).__init__()
        
        # gaussian

        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False).to(device)
        self.gaussian_filter.weight.data[:] = torch.from_numpy(gaussian_2D)

        # sobel
        self.device = device
        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False).to(device)
        self.sobel_filter_x.weight.data[:] = torch.from_numpy(sobel_2D)


        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False).to(device)
        self.sobel_filter_y.weight.data[:] = torch.from_numpy(sobel_2D.T)

       
    
    def forward(self, img):
        # set the setps tensors
        B, C, H, W = img.shape
        #blurred = torch.zeros((B, C, H, W)).to(self.device)
        #print('blurred:',blurred.is_leaf, blurred.grad_fn, blurred.requires_grad)
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        #print('grad_x:',grad_x.is_leaf, grad_x.grad_fn, grad_x.requires_grad)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        #grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
       

        # gaussian

        for c in range(C):
            #blurred[:, c:c+1] = self.gaussian_filter(img[:, c:c+1])
            grad_x = grad_x + self.sobel_filter_x(img[:, c:c+1])
            grad_y = grad_y + self.sobel_filter_y(img[:, c:c+1])
        #print('blurred:',blurred.is_leaf, blurred.grad_fn, blurred.requires_grad)
        #print('grad_x:',grad_x.is_leaf, grad_x.grad_fn, grad_x.requires_grad)
        # thick edges

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        

        return grad_x, grad_y