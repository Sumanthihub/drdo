# import package
import numpy as np
import torch
import torchvision
import time
import os
from torch.utils.data import DataLoader
from colorama import Style, Fore, Back
import argparse
from tqdm import tqdm

# import file
from derain_dehaze.utils.utils import *
from PIL import Image
import torchvision.transforms as transforms







@torch.no_grad()
def evaluate(model, loader,save_dir):
	
	print(Fore.GREEN + "==> Inference")

	start = time.time()
	model.eval()
	file_name= ''
	for image, image_name in tqdm(loader, desc='Inference'):
		print("******************************** image name",image_name)
		if torch.cuda.is_available():
			image = image.cuda()

		pred = model(image)   
		
		file_name = os.path.join(save_dir, image_name[0])
		torchvision.utils.save_image(pred.cpu(), file_name)
	
	# print('Costing time: {:.3f}'.format((time.time()-start)/60))
	# print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	# print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
	return file_name

def pred_detection(checkpoint,image_path,save_dir,model='derain_dehaze.models.MSBDN-RDFF.Net',dataset='derain_dehaze.utils.dataset.DatasetForInference'):
	# get the net and dataset function
 
	print("IMAGE_PATH",image_path)

	net_func = get_func(model)
	dataset_func = get_func(dataset)
	# print(Back.RED + 'Using Model: {}'.format(model) + Style.RESET_ALL)
	# print(Back.RED + 'Using Dataset: {}'.format(dataset) + Style.RESET_ALL)
	# print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# prepare the dataloader
	dataset = dataset_func(dir_path=image_path)
	# print(dataset.__get__,"this data set length")
	loader = DataLoader(dataset=dataset, num_workers=4, batch_size=1, drop_last=False, shuffle=False, pin_memory=True)
	print(Style.BRIGHT + Fore.YELLOW + "# Val data: {}".format(len(dataset)) + Style.RESET_ALL)
	# print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# prepare the model
	model = net_func()


	# load the checkpoint
	assert os.path.isfile(checkpoint), "The checkpoint '{}' does not exist".format(checkpoint)
	checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
	msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
	# print(Fore.GREEN + "Loaded checkpoint from '{}'".format(checkpoint) + Style.RESET_ALL)
	# print(Fore.GREEN + "{}".format(msg) + Style.RESET_ALL)
	# print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# move to GPU
	if torch.cuda.is_available():
		model = model.cuda()

	print("hello",save_dir)

	return evaluate(model, loader,save_dir)


