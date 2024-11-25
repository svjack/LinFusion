#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.pipelines.pipeline_highres_sdxl import StableDiffusionXLHighResPipeline
import torch

from src.linfusion import LinFusion
from src.tools import seed_everything



# In[2]:


model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
model_ckpt = "svjack/GenshinImpact_XL_Base"
device = torch.device('cuda')
pipe = StableDiffusionXLHighResPipeline.from_pretrained(
    model_ckpt, torch_dtype=torch.float16, 
    #variant='fp16'
).to(device)



# In[3]:


linfusion = LinFusion.construct_for(pipe,
                                   pretrained_model_name_or_path="Yuanshi/LinFusion-XL"
                                   )


# In[4]:


prompt = "An astronaut floating in space. Beautiful view of the stars and the universe in the background."
prompt = "solo,ZHONGLI\(genshin impact\),1boy,portrait,upper_body,highres,"
generator = torch.manual_seed(42)
pipe.enable_vae_tiling()
images = pipe(prompt,
              height=1024, width=2048, device=device,
              num_inference_steps=50, guidance_scale=7.5,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
              show_image=True, generator=generator, upscale_strength=0.32)
images[0]


# In[ ]:




