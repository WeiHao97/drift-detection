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


val_path = os.environ["IDD_val_path"] #"../IDD_Detection/val.txt"
annotations_path = os.environ["IDD_annotations_path"] #"../IDD_Detection/Annotations"
jpeg_path = os.environ["IDD_img_path"] #"../IDD_Detection/JPEGImages"

idd_root = os.environ["IDD_root"] #"../IDD_Detection"


label_dict = {1: 'bicycle',
			  2: 'bus',
			  3: 'car',
			  4: 'motorcycle',
			  5: 'rider',
			  6: 'traffic light',
			  7: 'traffic sign',
			  8: 'truck'}

label_dict = {label_dict[item]:item for item in label_dict}


def collate_fn(batch):
    return tuple(zip(*batch))


def _make_validation_dir():
	"""
	Move validation images into new directory
	"""

	txt_lst = [val_path]
	_val_json_lst = {} # map category to list 

	# for each object category
	for object_category in label_dict:   

		_json_lst = []
	    
		# iterate through sets
		# get all images that contain objects in category
		for idx in range(1):

			curr_txt = txt_lst[idx]

			with open(curr_txt, "r") as f:
				curr_filenames = f.readlines()
			
			for filename in curr_filenames:

				logging.info("curr filename: %s", filename)

				filename = filename.replace("\n", "")
				head, tail = os.path.split(filename)

				logging.info("head: %s, tail: %s", head, tail)

				curr_path = os.path.join(annotations_path, "%s.xml" % filename)

				logging.info("annotations path: %s", curr_path)

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

				logging.info("current jpeg file: %s", _json["filename"])
			    
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

					#make directories
					eval_annotation_path = os.path.join(os.path.join(idd_root, "eval-annotations"), head)
					if not os.path.exists(eval_annotation_path):
						os.system("mkdir -p %s" % eval_annotation_path)
					os.system("cp %s %s" % (curr_path, os.path.join(eval_annotation_path, "%s.xml" % tail)))

					eval_img_path = os.path.join(os.path.join(idd_root, "eval-img"), head)
					if not os.path.exists(eval_img_path):
						os.system("mkdir -p %s" % eval_img_path)
					os.system("cp %s %s" % (_json["filename"], os.path.join(eval_img_path, root.find("filename").text)))


		_val_json_lst[object_category] = _json_lst
		
		logging.info("moved images with %s, count %s", object_category, len(_json_lst))

	with open("idd-eval.json", 'w') as f:
		json.dump(_val_json_lst, f)

	logging.info("saved idd-eval.json")



if __name__ == "__main__":

	_make_validation_dir()


