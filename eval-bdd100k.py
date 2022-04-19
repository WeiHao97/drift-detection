import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json
import logging
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import warnings
from utils.torch.engine import train_one_epoch, evaluate
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.DEBUG)

##########################
# Define global variables
##########################


val_path = os.environ["VAL_PATH"]
val_json = os.environ["VAL_JSON"]


_label_dict = {1: 'bicycle',
			  2: 'bus',
			  3: 'car',
			  4: 'motorcycle',
			  5: 'rider',
			  6: 'traffic light',
			  7: 'traffic sign',
			  8: 'truck'}


label_dict = {label_dict[item]:item for item in label_dict}



class bdd100k_dataset(torch.utils.data.Dataset):
	

	def __init__(self, filenames, json_labels, path):
		self.filenames = filenames
		self.json_labels = json_labels
		self.path = path
		self.transform = transforms.Compose([transforms.ToTensor()])
											
	def __len__(self):
		return len(self.filenames)
	
	def __getitem__(self, idx):
		
		filename = os.path.join(self.path, self.filenames[idx])
		img = Image.open(filename).convert("RGB")
		
		assert(self.filenames[idx] == self.json_labels[idx]["name"])

		labels = self.json_labels[idx]["labels"] # list of dictionaries

		target = {}

		boxes = []
		categories = []
		crowded = []
		areas = []
		for item in labels:
			
			if item["category"] not in label_dict.keys():
				continue
			
			box2d = item["box2d"]
			
			x1 = box2d["x1"]
			y1 = box2d["y1"]
			
			x2 = box2d["x2"]
			y2 = box2d["y2"]
			
			boxes.append([x1, y1, x2, y2])
			categories.append(label_dict[item["category"]])
			crowded.append((False if "crowd" not in item["attributes"] else item["attributes"]["crowd"]))
			areas.append(abs(x2 - x1 + 1) * abs(y2 - y1 + 1))
		
		target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
		
		target["labels"] = torch.tensor(categories, dtype=torch.int64)
		target["image_id"] = torch.tensor([idx], dtype=torch.int64)
		target["area"] = torch.tensor(areas, dtype=torch.float32)
		target["iscrowd"] = torch.tensor(crowded, dtype=torch.uint8)
		
		return self.transform(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def eval(model, device):
	"""
	Get mAP scores for each object category
	"""
	with open(val_json) as file:
		val_labels = json.load(file)
	
	# for each object category
	for object_category in label_dict:   
	    ##################################
		    # Get all validation images and  #
		    # create dataloader              #
		    ##################################
		    
		filenames = []
		_json = []
	    
		copy_val_labels = copy.deepcopy(val_labels)
	    
		# get all images that contain objects in this category
		for idx in range(len(copy_val_labels)):
	        
			item = copy_val_labels[idx]
	        
			if "labels" not in item:
				continue
		        
			# if the image contains the object
			objects = item["labels"]
			objects = [obj for obj in objects if obj["category"] == object_category]
			if len(objects) > 0:
				filenames.append(item["name"])
				item["labels"] = objects
				_json.append(item)
	            
		assert len(filenames) == len(_json)
		print("Number of samples with %s: %s" % (object_category, len(filenames)))
		    
		dataset = bdd100k_dataset(filenames, 
	                              _json, 
	                              val_path)

		loader = DataLoader(dataset, batch_size=16,
	                        shuffle=True, 
	                        collate_fn=collate_fn)
	    
	    #######################
	    # evaluate on dataset #
	    #######################
	    
		evaluate(model, loader, device=device)



if __name__ == "__main__":

	# Instantiate pretrained model
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	# label 0 is reserved for the background class
	num_classes = 9

	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	model.load_state_dict(torch.load("models/faster-rcnn.pth"))

	logging.info("loaded trained model")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	
	eval(model, device)



