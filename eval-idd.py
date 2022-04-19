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
import xml.etree.ElementTree as ET
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


image_path = os.environ["IMAGE_PATH"] #"/local/rcs/lnh2116/eval-img"


label_dict = {1: 'bicycle',
			  2: 'bus',
			  3: 'car',
			  4: 'motorcycle',
			  5: 'rider',
			  6: 'traffic light',
			  7: 'traffic sign',
			  8: 'truck'}

label_dict = {label_dict[item]:item for item in label_dict}



class idd_dataset(torch.utils.data.Dataset):
	

	def __init__(self, json_lst):

		self.json_lst = json_lst
		self.transform = transforms.Compose([transforms.ToTensor()])
											
	def __len__(self):
		return len(self.json_lst)

	def __getitem__(self, idx):
		
		_json = self.json_lst[idx]
		filename = image_path + _json["filename"].split("JPEGImages")[1]
		img = Image.open(filename).convert("RGB")

		objects = _json["objects"]

		target = {}

		boxes = []
		categories = []
		crowded = []
		areas = []
		for item in objects:
			
			bbox = item["bbox"]
			x1 = bbox["xmin"]
			y1 = bbox["ymin"]
			x2 = bbox["xmax"]
			y2 = bbox["ymax"]
			boxes.append([x1, y1, x2, y2])

			categories.append(label_dict[item["category"]])

			crowded.append(False)
			
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

	with open("idd-eval.json") as f:
		_idd_val_json = json.load(f)
	

	# for each object category
	for object_category in _idd_val_json:   


		_json_lst = _idd_val_json[object_category]
	    
		logging.info("evaluating on images with %s, count %s", object_category, len(_json_lst))

		dataset = idd_dataset(_json_lst)

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

	model.load_state_dict(torch.load("models/faster-rcnn.pth", map_location=torch.device("cpu")))

	model = nn.DataParallel(model)

	logging.info("loaded trained model")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	eval(model, device)


