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
from engine import train_one_epoch, evaluate
from shapely.geometry import box as sbox
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.DEBUG)

##########################
# Define global variables
##########################


train_path = "bdd100k/images/100k/train"
val_path = "/local/rcs/lnh2116/val"

train_json = "bdd100k/images/100k/labels/det_train.json"
val_json = "/local/rcs/lnh2116/det_val.json"


label_dict = {1: 'bicycle',
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


def _compute_f1(model, loader):
	"""
	Compute f1 score for entire dataset
	with 0.5 confidence score threshold
	"""
	
	_FP = 0
	_TP = 0
	_FN = 0

	# For each image, we have ground truth boxes and 
		# predicted boxes (number of predictions, maybe more or less)

	batch_num = 0
	for x, y in loader:

		logging.info("evaluating batch num %s", batch_num)

		x = list(img.to(f'cuda:{model.device_ids[0]}') for img in x)

		output = model(x)

		# iterate through each image
		for idx in range(len(output)):

			predicted_boxes = output[idx]["boxes"]
			box_scores = output[idx]["scores"]
			box_labels = output[idx]["labels"]

			matched_ground_truth = {(box[0].item(), box[1].item(), box[2].item(), box[3].item()):False for box in y[idx]["boxes"]}

			# iterate through each predicted box
			for _idx in range(len(predicted_boxes)):

				# if there are no more unmatched ground truth boxes 
				# this is a FP. Continue to next step
				if len(matched_ground_truth) == 0:
					logging.info("FOUND FP: no more unmatched ground truth boxes")
					_FP += 1
					continue

				pred = predicted_boxes[_idx]
				pred = (pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item())
				score = box_scores[_idx]
				label = box_labels[_idx]

				# if the confidence score is less than 0.2, we ignore it
				if score < 0.2:
					continue


				# Get all of the ground truth boxes with 
				# the predicted label that haven't been matched

				possible_ground_truth_boxes = [y[idx]["boxes"][i] for i in range(len(y[idx]["boxes"])) if y[idx]["labels"][i] == label]
				possible_ground_truth_boxes = [(box[0].item(), box[1].item(), box[2].item(), box[3].item()) for box in possible_ground_truth_boxes if not matched_ground_truth[(box[0].item(), box[1].item(), box[2].item(), box[3].item())]]
				
				# If there are none, this prediction is a FP
				if len(possible_ground_truth_boxes) == 0:
					logging.info("FOUND FP: no ground truth boxes for this label")
					_FP += 1
					continue

				
				# try to find a match
				box_max_IoU = possible_ground_truth_boxes[0]
				max_IoU = _compute_IoU(box_max_IoU, pred)
				for possible_gtb in possible_ground_truth_boxes[1:]:

					curr_IoU = _compute_IoU(possible_gtb, pred)
					
					if curr_IoU > max_IoU:
						max_IoU = curr_IoU
						box_max_IoU = possible_gtb

				
				# if the predicted box doesn't intersect with any of the ground truh boxes,
				# it is a FP and we don't count this as a match
				if max_IoU == 0:
					logging.info("FOUND FP: no intersection with any ground truth boxes")
					_FP += 1
					continue
				

				# record matched ground truth box
				matched_ground_truth[box_max_IoU] = True


				# if IoU is at least 0.3, it is a TP
				if max_IoU >= 0.30:
					logging.info("FOUND TP: IoU %s", max_IoU)
					_TP += 1
					continue

				else:
					logging.info("FOUND FP: IoU %s", max_IoU)
					_FP += 1
					continue

			# look at all gound truth boxes that weren't matched
			# they are all FN's
			logging.info("ADDING FN: %s", len([1 for box in matched_ground_truth if matched_ground_truth[box] == False]))
			_FN += len([1 for box in matched_ground_truth if matched_ground_truth[box] == False])

		logging.info("current confusion TP/FP/FN: %s/%s/%s", _TP, _FP, _FN)
		batch_num += 1


	# return F1 score
	precision = _TP / (_TP + _FP)
	recall = _TP / (_TP + _FN)
	return (2 * precision * recall) / (precision + recall)



def _compute_IoU(box1, box2):
	"""
	Compute intersection over union
	"""	
	b1 = sbox(box1[0], box1[1], box1[2], box1[3])
	b2 = sbox(box2[0], box2[1], box2[2], box2[3])

	return b1.intersection(b2).area / b1.union(b2).area


def eval(model):
	"""
	Get mAP scores for each object category
	"""
	with open(val_json) as file:
		val_labels = json.load(file)

	filenames = []
	_json = []
	
	# for each object category
	for object_category in label_dict:
 
		
		##################################
		# Get all validation images and  #
		# create dataloader              #
		##################################
		
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

	logging.info("collected %s images", len(filenames))
	
	dataset = bdd100k_dataset(filenames, 
							  _json, 
							  val_path)

	loader = DataLoader(dataset, batch_size=16,
						shuffle=True, 
						collate_fn=collate_fn)
		
	#######################
	# compute f1 score #
	#######################
		
	f1_score = _compute_f1(model, loader)

	logging.info("F1 Score: %s", f1_score)



if __name__ == "__main__":

	# Instantiate pretrained model
	backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101',pretrained=True)
	anchor_generator = AnchorGenerator(sizes=((64,), (128,), (256,), (512,), (1024,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)

	# label 0 is reserved for the background class
	num_classes = 9

	model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
		
	# use all available GPUs
	model = nn.DataParallel(model, device_ids=[0, 1, 3, 2, 4, 5, 6, 7])

	model.load_state_dict(torch.load("models/faster-rcnn-resnet101.pth")["model"])
	model.to("cuda:0")

	logging.info("loaded trained model")

	model.eval()
	eval(model)

