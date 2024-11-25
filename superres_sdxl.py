#!/usr/bin/env python
# coding: utf-8

# In[1]:


from src.pipelines.pipeline_superres_sdxl import StableDiffusionXLSuperResPipeline
from diffusers import AutoPipelineForText2Image
import torch


from src.linfusion import LinFusion


# In[2]:


model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
model_ckpt = "svjack/GenshinImpact_XL_Base"
device = torch.device('cuda')


pipe = AutoPipelineForText2Image.from_pretrained(
    model_ckpt, torch_dtype=torch.float16, 
    #variant="fp16"
).to(device)


# In[3]:


prompt = "An astronaut floating in space. Beautiful view of the stars and the universe in the background."
prompt = "solo,ZHONGLI\(genshin impact\),1boy,portrait,upper_body,highres,"
generator = torch.manual_seed(123)
image = pipe(
    prompt, generator=generator
).images[0]



# In[4]:


image


# In[6]:


pipe = StableDiffusionXLSuperResPipeline.from_pretrained(
    model_ckpt, torch_dtype=torch.float16, 
    #variant="fp16"
).to(device)




# In[7]:


linfusion = LinFusion.construct_for(pipe,
                                   pretrained_model_name_or_path="Yuanshi/LinFusion-XL"
                                   )




# In[8]:


generator = torch.manual_seed(123)
pipe.enable_vae_tiling()
image = pipe(image=image, prompt=prompt,
             height=2048, width=2048, device=device, 
             num_inference_steps=50, guidance_scale=7.5,
             cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, gaussian_sigma=0.8,
             generator=generator, upscale_strength=0.32).images[0]




# In[9]:


image



# In[ ]:




