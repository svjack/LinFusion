#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.pipelines.pipeline_superres_sdxl import StableDiffusionXLSuperResPipeline
from diffusers import AutoPipelineForText2Image
import torch


from src.tools import (
    forward_unet_wrapper, 
    forward_resnet_wrapper, 
    forward_crossattndownblock2d_wrapper, 
    forward_crossattnupblock2d_wrapper,
    forward_downblock2d_wrapper, 
    forward_upblock2d_wrapper,
    forward_transformer_block_wrapper)
from src.linfusion import LinFusion


# In[3]:


model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
model_ckpt = "svjack/GenshinImpact_XL_Base"
device = torch.device('cuda')


pipe = AutoPipelineForText2Image.from_pretrained(
    model_ckpt, torch_dtype=torch.float16, 
    #variant="fp16"
).to(device)


# In[4]:


prompt = "An astronaut floating in space. Beautiful view of the stars and the universe in the background."
prompt = "solo,ZHONGLI\(genshin impact\),1boy,portrait,upper_body,highres,"
generator = torch.manual_seed(0)
image = pipe(
    prompt, height=512, width=1024, generator=generator
).images[0]
image


# In[5]:


pipe = StableDiffusionXLSuperResPipeline.from_pretrained(
    model_ckpt, torch_dtype=torch.float16, 
    #variant="fp16"
).to(device)


# In[7]:


linfusion = LinFusion.construct_for(pipe,
                                   pretrained_model_name_or_path="Yuanshi/LinFusion-XL"
                                   )
pipe.enable_vae_tiling()




# In[8]:


generator = torch.manual_seed(0)
image = pipe(image=image, prompt=prompt,
             height=1024, width=2048, device=device, 
             num_inference_steps=50, guidance_scale=7.5,
             cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
             generator=generator, upscale_strength=0.32).images[0]
image



# In[9]:


generator = torch.manual_seed(0)
image = pipe(image=image, prompt=prompt,
             height=2048, width=4096, device=device, 
             num_inference_steps=50, guidance_scale=7.5,
             cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
             generator=generator, upscale_strength=0.24).images[0]
image



# In[10]:


for _, _module in pipe.unet.named_modules():
    if _module.__class__.__name__ == 'BasicTransformerBlock':
        _module.set_chunk_feed_forward(16, 1)
        _module.forward = forward_transformer_block_wrapper(_module)
    elif _module.__class__.__name__ == 'ResnetBlock2D':
        _module.nonlinearity.inplace = True
        _module.forward = forward_resnet_wrapper(_module)
    elif _module.__class__.__name__ == 'CrossAttnDownBlock2D':
        _module.forward = forward_crossattndownblock2d_wrapper(_module)
    elif _module.__class__.__name__ == 'DownBlock2D':
        _module.forward = forward_downblock2d_wrapper(_module)
    elif _module.__class__.__name__ == 'CrossAttnUpBlock2D':
        _module.forward = forward_crossattnupblock2d_wrapper(_module)
    elif _module.__class__.__name__ == 'UpBlock2D':
        _module.forward = forward_upblock2d_wrapper(_module)   

pipe.unet.forward = forward_unet_wrapper(pipe.unet)



# In[11]:


generator = torch.manual_seed(0)
image = pipe(image=image, prompt=prompt,
             height=4096, width=8192, device=device, 
             num_inference_steps=50, guidance_scale=7.5,
             cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
             generator=generator, upscale_strength=0.16).images[0]
image



# In[12]:


generator = torch.manual_seed(0)
image = pipe(image=image, prompt=prompt,
             height=8192, width=16384, device=device, 
             num_inference_steps=50, guidance_scale=7.5,
             cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
             generator=generator, upscale_strength=0.08).images[0]
image



# In[13]:


image.size


# In[ ]:




