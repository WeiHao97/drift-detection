import argparse
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


label_dict = {1: 'bicycle',
			  2: 'bus',
			  3: 'car',
			  4: 'motorcycle',
			  5: 'rider',
			  6: 'traffic light',
			  7: 'traffic sign',
			  8: 'truck'}

label_dict = {label_dict[item]:item for item in label_dict}

cityscapes_dir = ""
img_dir =  ""


class foggy_cityscapes_dataset(torch.utils.data.Dataset):
	
	def __init__(self, json_lst):
		
		self.json_lst = json_lst
		self.transform = transforms.Compose([transforms.ToTensor()])
		
	def __len__(self):
		return len(self.json_lst)
	
	def __getitem__(self, idx):
		
		_json = self.json_lst[idx]
		file_path = _json["file_path"]
		img = Image.open(file_path).convert("RGB")
		
		objects = _json["objects"]
		
		target = {}
		
		boxes = []
		categories = []
		crowded = []
		areas = []
		for item in objects:
			
			x1 = item["xmin"]
			y1 = item["ymin"]
			x2 = item["xmax"]
			y2 = item["ymax"]
			boxes.append([x1, y1, x2, y2])
			
			categories.append(label_dict[item["label"]])
			
			crowded.append(False)
			
			areas.append(abs(x2 - x1 + 1) * abs(y2 - y1 + 1))
			
		target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
		target["labels"] = torch.tensor(categories, dtype=torch.int64)
		target["image_id"] = torch.tensor([idx], dtype=torch.int64)
		target["area"] = torch.tensor(areas, dtype=torch.float32)
		target["iscrowd"] = torch.tensor(crowded, dtype=torch.uint8)

		return self.transform(img), target


def _process_polygon(polygon, height, width):
	"""
	Given a polygon, return xmin, ymin, xmax, and ymax
	"""
	xmin = width
	xmax = 0
	ymin = height
	ymax = 0
	for coord in polygon:
		
		coord_x, coord_y = coord
		
		xmin = min(xmin, coord_x)
		xmax = max(xmax, coord_x)
		
		ymin = min(ymin, coord_y)
		ymax = max(ymax, coord_y)
	
	return xmin, ymin, xmax, ymax


def collate_fn(batch):
	return tuple(zip(*batch))


def eval(model, device):
	"""
	Get mAP scores for each object category
	"""
	
	# for each object category
	for object_category in label_dict:
		
		_json_lst = []

		sets = {"cityscapes_train":"foggy-train", 
		        "cityscapes_val":"foggy-val"}

		for _set in sets:
		
			# path to annotations
			set_path = os.path.join(cityscapes_dir, _set)

			# path to images
			img_path = os.path.join(img_dir, sets[_set])

			for sub_folder in os.listdir(set_path):
				
				sub_path = os.path.join(set_path, sub_folder)
				curr_img_path = os.path.join(img_path, sub_folder)
				
				for file in os.listdir(sub_path):
					
					if not file.endswith(".json"):
						continue
					
					file_path = os.path.join(sub_path, file)
					
					with open(file_path) as f:
						_curr_json = json.load(f)
					
					filtered_obj_lst = []
					for obj in _curr_json["objects"]:
						
						if obj["label"] != object_category:
							continue
						
						datum = {}
						datum["label"] = object_category
						
						xmin, ymin, xmax, ymax = _process_polygon(
																  obj["polygon"],
																  _curr_json["imgHeight"],
																  _curr_json["imgWidth"]
																 )
						datum["xmin"] = xmin
						datum["ymin"] = ymin
						datum["xmax"] = xmax
						datum["ymax"] = ymax

						filtered_obj_lst.append(datum)
					
					if len(filtered_obj_lst) == 0:
						continue
					
					data = {}
					data["imgHeight"] = _curr_json["imgHeight"]
					data["imgWidth"] = _curr_json["imgWidth"]
					data["file_path"] = os.path.join(curr_img_path, file.replace("_gtFine_polygons.json", "_leftImg8bit_transmittance_beta_0.02.png"))
					data["objects"] = filtered_obj_lst

					if not os.path.exists(data["file_path"]):
						continue
					
					_json_lst.append(data)
		
		
		
		logging.info("evaluating on %s images with category %s", 
					 len(_json_lst), 
					 object_category)
		
		dataset = foggy_cityscapes_dataset(_json_lst)
		
		loader = DataLoader(dataset, batch_size=16,
		                    shuffle=True, 
		                    collate_fn=collate_fn)
   
		#######################
		# evaluate on dataset #
		#######################
	
		evaluate(model, loader, device=device)
				

def _get_args():
	"""
	Get cityscapes_dir and img_dir
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--paths", nargs=2)
	args = parser.parse_args()
	return args.paths


if __name__ == "__main__":

	cityscapes_dir, img_dir = _get_args()

	logging.info("parsed cityscapes_dir %s img_dir %s", cityscapes_dir, img_dir)

	# Instantiate pretrained model
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	# label 0 is reserved for the background class
	num_classes = 9

	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	device = torch.device('cuda:0')

	model.load_state_dict(torch.load("models/faster-rcnn.pth", map_location=device))

	model = nn.DataParallel(model, device_ids=[0, 1, 3, 2, 4, 5, 6, 7])

	logging.info("loaded trained model")

	model.to(device)

	model.eval()
	eval(model, device)


