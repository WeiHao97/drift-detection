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
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

warnings.filterwarnings('ignore')

logging.getLogger().setLevel(logging.DEBUG)

##########################
# Define global variables
##########################


train_path = "../bdd100k/images/100k/train"
val_path = "../bdd100k/images/100k/val"

train_json = "../bdd100k/images/100k/labels/det_train.json"
val_json = "../bdd100k/images/100k/labels/det_val.json"


label_dict = {
                1: "pedestrian",
                2: "rider",
                3: "car",
                4: "truck",
                5: "bus",
                6: "train",
                7: "motorcycle",
                8: "bicycle",
                9: "traffic light",
                10: "traffic sign"
             }

label_dict = {label_dict[item]:item for item in label_dict}

weather_domains = ["rainy", "snowy", "clear", "overcast", "partly cloudy", "foggy"]


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
            areas.append(abs(x1 - x2) * abs(y1 - y2))
        
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        
        target["labels"] = torch.tensor(categories, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)
        target["area"] = torch.tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.tensor(crowded, dtype=torch.uint8)
        
        return self.transform(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_model(model, dataloader,
                optimizer, scheduler, 
                num_epochs, device):
    
    # switch to training mode
    model.train()
    
    # send params to device
    model = model.to(device)

    for epoch in range(num_epochs):
        
        logging.info("train_model --- epoch: %s" % epoch)

        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, 
        	            dataloader, device,
                        epoch, print_freq=10)
        
        # update learning rate
        scheduler.step()

    return model


def get_clear_model():
	"""
	Train model on clear domain and save it
	"""

	with open(train_json) as file:
		train_labels = json.load(file)

	trainX_filenames = []
	_train_json = []

	# get all images whose weather attribute is clear
	for item in train_labels:

		if "labels" not in item:
			continue

	    # if the weather is clear
		if item["attributes"]["weather"] == "clear":

			trainX_filenames.append(item["name"])
			_train_json.append(item)

	assert len(trainX_filenames) == len(_train_json)
	logging.info("Number of `clear` training samples: %s" % len(trainX_filenames))

	dataset = bdd100k_dataset(trainX_filenames, 
	                          _train_json, 
	                          train_path)
	
	loader = DataLoader(dataset, batch_size=16,
	                    shuffle=True, 
	                    collate_fn=collate_fn)

	# Instantiate pretrained model
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	# label 0 is reserved for the background class
	num_classes = 11

	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	# Train the model on "clear" images

	# Observe that all parameters are being optimized
	optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

	# Decay LR by a factor of 0.1 every 2 epochs
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

	num_epochs = 5

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_model(model, loader,
	            optimizer, 
	            scheduler, 
	            num_epochs, device)

	torch.save(model.state_dict(), "models/faster-rcnn-clear.pth")

	logging.info("saved trained model")

	return model



if __name__ == "__main__":

	model = get_clear_model()



