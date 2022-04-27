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
from utils.torch.engine import train_one_epoch, evaluate
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.DEBUG)


labels = {"human--rider--motorcyclist":"rider",
          "human--rider--bicyclist":"rider",
          "human--rider--other-rider":"rider",
          "object--vehicle--car":"car",
          "object--vehicle--truck": "truck",
         "object--vehicle--bus": "bus",
         "object--vehicle--motorcycle": "motorcycle",
         "object--vehicle--bicycle": "bicycle",
          "object--traffic-sign--front": "traffic sign",
          "object--traffic-sign--back": "traffic sign"
         }

label_prefix = {"object--traffic-light": "traffic light"}



label_dict = {1: 'bicycle',
			  2: 'bus',
			  3: 'car',
			  4: 'motorcycle',
			  5: 'rider',
			  6: 'traffic light',
			  7: 'traffic sign',
			  8: 'truck'}

label_dict = {label_dict[item]:item for item in label_dict}



class mapillary_dataset(torch.utils.data.Dataset):
	

	def __init__(self, filenames, json_lst):
		self.filenames = filenames
		self.json_lst = json_lst
		self.transform = transforms.Compose([transforms.ToTensor()])
											
	def __len__(self):
		return len(self.filenames)

	def get_bounding_box(self, polygon, width, height):
		"""
		Given a polygon, return bounding box
		parametrized by xmin, ymin, xmax, & ymax
		"""
		xmin = width
		xmax = 0
		ymin = height
		ymax = 0
		for coord in polygon:
			xmin = min(xmin, coord[0])
			xmax = max(xmax, coord[0])
			ymin = min(ymin, coord[1])
			ymax = max(ymax, coord[1])
		return xmin, ymin, xmax, ymax

	def get_category(self, label):
		"""
		Return bdd100k numerical label in [1, 8]
		"""
		if label in labels:
			return label_dict[labels[label]]

		assert label.startswith("object--traffic-light")
		return label_dict["traffic light"]

	
	def __getitem__(self, idx):
		
		filename = self.filenames[idx]
		img = Image.open(filename).convert("RGB")

		objects = self.json_lst[idx]["objects"]

		target = {}

		boxes = []
		categories = []
		crowded = []
		areas = []
		for item in objects:
			
			x1, y1, x2, y2 = self.get_bounding_box(item["polygon"], 
				                                   self.json_lst[idx]["width"], 
				                                   self.json_lst[idx]["height"])
			
			boxes.append([x1, y1, x2, y2])

			categories.append(self.get_category(item["label"]))

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


def _label_match(label, object_category):
	"""
	Returns True if label belongs to object_category
	"""
	if label in labels and labels[label] == object_category:
		return True
	if label.startswith("object--traffic-light") and object_category == "traffic light":
		return True

	return False


def eval(model, device, val_path, val_json):
	"""
	Get mAP scores for each object category
	"""

	json_paths = [val_json]
	image_paths = [val_path]

	# for each object category
	for object_category in label_dict:   

		filenames = []
		_json_lst = []
	    
		
		# iterate through training and validation sets
		# get all images that contain objects in category
		for idx in range(1):

			curr_json_path = json_paths[idx]
			
			for json_file in os.listdir(curr_json_path):

				with open(os.path.join(curr_json_path, json_file), 'r') as f:
					_json = json.load(f)

				objects = _json["objects"]
				objects = [obj for obj in objects if _label_match(obj["label"], object_category)]
				if len(objects) > 0:
					_json["objects"] = objects
					_json_lst.append(_json)
					filenames.append(os.path.join(image_paths[idx], json_file.replace(".json", ".jpg")))

		assert len(filenames) == len(_json_lst)
		logging.info("evaluating images with %s, count: %s", object_category, len(filenames))

		dataset = mapillary_dataset(filenames, _json_lst)

		loader = DataLoader(dataset, batch_size=16,
	                        shuffle=True, 
	                        collate_fn=collate_fn)
	    
	    #######################
	    # evaluate on dataset #
	    #######################
	    
		evaluate(model, loader, device=device)


def _get_args():
	"""
	Return val_path and val_json
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument("--paths", nargs=2)
	args = parser.parse_args()
	return args.paths


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

	val_path, val_json = _get_args()

	logging.info("parsed val_path %s val_json %s", val_path, val_json)

	eval(model, device, val_path, val_json)




