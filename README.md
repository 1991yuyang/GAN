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
epoch:number of training iterations  
batch_size: sample batch size of one training step 
g_init_lr: initial learning rate of generator 
g_final_lr: final learning rate of generator  
d_init_lr: initial learning rate of discriminator  
d_final_lr: final learning rate of discriminator  
noise_dim: dimension of noise, if use_cnn is True, shape of noise will be [noise_dim, 1, 1], if use_cnn is False, shape of noise will be [noise_dim]  
