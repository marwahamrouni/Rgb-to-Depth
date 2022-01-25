
# Extracting depth maps of single RGB images using GANs
A supervised and an unsupervised models for predicting depth information implemented in PyTorch. The generator model  has an encoder-decoder architecture; the pre-traned EfficientNet-B5 is exploited as backbone of the encoder to to extract the features essential for reconstruction the depth map, while the decoder is based on residual learning . We tested the performance of our models on the NYU Depth V2 Dataset (Eigen Split) and the KITTI Dataset (Eigen Split). 

## Requirements

* Python 3
* PyTorch 
  * Tested with PyTorch 1.7.1
* CUDA 11.0 (if using CUDA)
## Example of results
### NYU Depth V2
![alt text](https://github.com/marwahamrouni/RGBtoDEPTH/blob/master/Result%20example/result_example_nyu.png)
### KITTI
![alt text](https://github.com/marwahamrouni/RGBtoDEPTH/blob/master/Result%20example/result_example_kitti.png)

