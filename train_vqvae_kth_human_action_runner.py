#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../video')
from train_vqvae_video import train
from dataset import lmdb_kth_running
sys.path.append('../image')
from vqvae import VQVAE
from torchvision import utils
import torch
import matplotlib.pyplot as plt


# In[2]:


lr = 0.0001
device = 'cuda'
epoch_num = 400
batch_size = 64
run_num = 1
image_samples = 10
model = VQVAE(in_channel=1,
                channel=32,
                n_res_block=4,
                n_res_channel=256,
                embed_dim=32,
                n_embed=1024,
                decay=0.99, )


# In[3]:


videos_dir  = '../video/datasets/kth/kth_human_actions/running'
kthpath = "../video/datasets/kth/"
lmdb_path = kthpath + 'kth_running_lmdb'


# In[4]:


out_image = []


# In[5]:


def callback(sample, out, epoch):
    print(sample.shape)
    print(out.shape)
    out = out + 0.5
    sample = sample + 0.5
    global out_image
    out = out[:,0,:,:]
    out_image = out
    sample = sample[:,0,:,:]
    fig=plt.figure(figsize=(8, 8))

    print(out.shape)
    print(sample.shape)
    
    for i in range (3):
        fig.add_subplot(3, 2, i*2+1)
        plt.imshow(out[4+i ,:,:].detach().cpu().numpy() ,'gray')
        fig.add_subplot(3, 2, i*2+2)
        plt.imshow(sample[4+i,:,:].detach().cpu().numpy(),'gray')
    plt.show()


# In[6]:


dataset = lmdb_kth_running(lmdb_path, 1)


# In[ ]:



train(dataset, model, epoch_num, batch_size, lr, device, run_num, image_samples, callback)


# In[ ]:


od = out_image.detach().cpu().numpy()
# for i in range(10):
#     plt.imshow(od[i,:,:].detach().cpu().numpy()/256)
#     plt.show()


# In[ ]:


od[1,:,:]


