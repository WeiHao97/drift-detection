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
from engine import train_one_epoch, evaluate
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.DEBUG)

##########################
# Define global variables
##########################


train_path = "../IDD_Detection/train.txt"
val_path = "../IDD_Detection/val.txt"
test_path = "../IDD_Detection/test.txt"

annotations_path = "../IDD_Detection/Annotations"
jpeg_path = "../IDD_Detection/JPEGImages"


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
		filename = _json["filename"]
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

	txt_lst = [train_path, val_path, test_path]

	# for each object category
	for object_category in label_dict:   

		_json_lst = []
	    
		# iterate through training, validation, and test sets
		# get all images that contain objects in category
		for idx in range(3):

			curr_txt = txt_lst[idx]

			with open(curr_txt, "r") as f:
				curr_filenames = f.readlines()
			
			for filename in curr_filenames:

				filename = filename.replace("\n", "")
				head, tail = os.path.split(filename)

				curr_path = os.path.join(annotations_path, "%s.xml" % filename)

				if not os.path.exists(curr_path):
					continue
			    
				tree = ET.parse(curr_path)
				root = tree.getroot()
			    
				_json = {

							"filename": os.path.join(os.path.join(jpeg_path, head), root.find("filename").text),
							"width": float(root.find("size").find("width").text),
							"height": float(root.find("size").find("height").text),
			             
			             }
			    
				if not os.path.exists(_json["filename"]):
					continue
			    
				obj_lst = []
				objects = root.findall("object")
				for obj in objects:
					if obj.find("name").text == object_category:
						curr_obj = {}
						curr_obj["category"] = obj.find("name").text
						curr_obj["bbox"] = {"xmin": float(obj.find("bndbox").find("xmin").text),
			                               "xmax": float(obj.find("bndbox").find("xmax").text),
			                               "ymin": float(obj.find("bndbox").find("ymin").text),
			                               "ymax": float(obj.find("bndbox").find("ymax").text)}
						obj_lst.append(curr_obj)

				_json["objects"] = obj_lst
				
				if len(obj_lst) > 0:
					_json_lst.append(_json)


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

	model.load_state_dict(torch.load("models/faster-rcnn.pth"))

	logging.info("loaded trained model")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	#device = torch.device('cpu')
	eval(model, device)


