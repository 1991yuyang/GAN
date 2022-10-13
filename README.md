# GAN
![](https://github.com/1991yuyang/GAN/blob/main/train_process.gif)
## Data Preparation  
data_root_dir  
&nbsp;&nbsp;&nbsp;&nbsp;train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;......  
&nbsp;&nbsp;&nbsp;&nbsp;val  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.jpg  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;......  
## Hyper Parameters  
epoch: number of training iterations  
batch_size: sample batch size of one training step 
g_init_lr: initial learning rate of generator 
g_final_lr: final learning rate of generator  
d_init_lr: initial learning rate of discriminator  
d_final_lr: final learning rate of discriminator  
noise_dim: dimension of noise, if use_cnn is True, shape of noise will be [noise_dim, 1, 1], if use_cnn is False, shape of noise will be [noise_dim]  
d_train_times: train the generator every d_train_times times the discriminator is trained  
data_root_dir: data root directory  
num_workers: num_workers of pytorch data loader  
img_size: size of image generated from generator, if use_cnn is True, img_size will be 2 ** len(generator_features), len(generator_features) is length of generator_features  
save_img_total_step: perform image generation every save_img_total_step steps of training and save the image  
img_count: the number of images generated each time  
discriminator_weight_min: min value of weight of discriminator
discriminator_weight_max: max  value of weight of discriminator  
use_cnn: if specified as True, the convolution neural network is used, otherwise the MLP is used  
generator_features: channels of every layer of generator if use_cnn is True, number of neurons of every layer if use_cnn is False  
img_save_dir: storage directory for generated images
