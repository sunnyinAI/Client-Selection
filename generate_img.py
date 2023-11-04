from diffusers import DDPMScheduler, UNet2DModel
import torch
from PIL import Image
import numpy as np
import os
import glob
import random
import sys
from tqdm import tqdm
import math
def mk(name):
	if not os.path.exists(name):
		os.makedirs(name,exist_ok=True)
############################################
root='/home/sunny/Client/karmany/P_'
cls='P'
batch=32
total_images=3500
############################################
if len(sys.argv)>1:
	root = sys.argv[-1]
# print(sys.argv)
scheduler = DDPMScheduler.from_pretrained(f'{root}/scheduler')
model = UNet2DModel.from_pretrained(f'{root}/unet').to("cuda")
scheduler.set_timesteps(100)
sample_size = model.config.sample_size
mk(f'Pn/train/{cls}')
counter = 0
for _ in tqdm(range(math.ceil(total_images/batch))):
	if counter >= total_images:
		break
	input = torch.randn((batch, 3, sample_size, sample_size)).to("cuda")
	for t in scheduler.timesteps:
		with torch.no_grad():
			noisy_residual = model(input, t).sample
			input = scheduler.step(noisy_residual, t, input).prev_sample
	image = (input / 2 + 0.5).clamp(0, 1) * 255
	image = image.cpu().permute(0, 2, 3, 1).numpy().astype("uint8")
	for i, im in enumerate(image):
		if counter >= total_images:
			print("finished generating images")
			break
		im = Image.fromarray(im)
		im.save(f'./Pn/train/{cls}/{str(counter)}.jpeg')
		counter += 1
	for i,im in enumerate(image): 
		
		if counter>=total_images:
			print("finished generating images")
			break
		im = Image.fromarray(im)
		im.save(f'./Pn/train/{cls}/{str(counter)}.jpeg')
		
		counter+=1
	# print(f'{str(counter)} images geneerated \r')

# images=[str(x)+'.jpeg' for x in range(5000)]
# for x in range(1,5):
# 	print(f'starting {str(x)}k')
# 	mk(f'{str(x)}k/train/{cls}')
# 	random.shuffle(images)
# 	for i,y in enumerate(images):
# 		if i==x*1000:
# 			break
# 		os.system(f'cp ./5k/train/{cls}/{y} ./{str(x)}k/train/{cls}/{y}')
print("finished")
