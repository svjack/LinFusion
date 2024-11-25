#!/usr/bin/env python
# coding: utf-8

# In[1]:


from diffusers import AutoPipelineForText2Image
import torch
from src.linfusion import LinFusion


from src.tools import seed_everything


# In[2]:


#sd_repo = "stabilityai/stable-diffusion-xl-base-1.0"
sd_repo = "svjack/GenshinImpact_XL_Base"
pipeline = AutoPipelineForText2Image.from_pretrained(
    sd_repo, torch_dtype=torch.float16, 
    #variant="fp16"
).to(torch.device("cuda"))


# In[3]:


linfusion = LinFusion.construct_for(pipeline, 
                                    pretrained_model_name_or_path="Yuanshi/LinFusion-XL")


# In[4]:


seed_everything(123)
prompt = "solo,ZHONGLI\(genshin impact\),1boy,portrait,upper_body,highres,"

image = pipeline(
    prompt
).images[0]
#image


# In[5]:


image.save("basic_zhongli.png")



# In[6]:


from IPython import display


# In[7]:


display.Image("basic_zhongli.png", width= 512, height=
             512
             )


# In[ ]:




