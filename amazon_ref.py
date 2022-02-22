import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import json
import time
import copy
import logging
import warnings

warnings.filterwarnings('ignore')

from drift_detection import *
from utils.model_utils import *
from utils.dataset_utils import *
from utils.stats import *

logging.getLogger().setLevel(logging.DEBUG)

data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([224, 224]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
}

batch_sizes = [2 ** x for x in range(0, 8)]



def load_model(filename):
	"""
	Load trained model in evaluation mode
	"""
	model = torchvision.models.mobilenet_v2(pretrained=True)
	num_ftrs = model.classifier[1].in_features
	# Here the size of each output sample is set to 2.
	# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
	model.classifier[1] = nn.Linear(num_ftrs, 31)
	model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
	model.eval()
	return model


def compute_logits(model, device, s_loader):
	"""
	Compute logits (to be given as input to drift detector)
	"""

	flag = True
	ref_logist = 0
	for inputs, labels in s_loader['val']:
	    inputs = inputs.to(device)
	    with torch.no_grad():
	        outputs = model(inputs).cpu()
	    torch.cuda.empty_cache()
	    if flag:
	        ref_logist = outputs
	        flag = False
	    else:
	        ref_logist = torch.cat((ref_logist,outputs), 0)

	return ref_logist


"""
- No online training
- Use batch sizes of 2^x for x in [0, 7]
- For each batch size,
    - Have table of FP/TP/TN/FN
		- If accuracy drops by 10%, then it’s a positive (drift) regardless of the actual domain
- Have a plot where x is batch size and y is time in milliseconds
- Want avg/std/min/max of KS statistics for each domain (should get one per image)
"""




if __name__ == '__main__':

	batch_size = 20

	# load label dictionary
	with open("models/label_dict.json", "r") as infile:
	    label_dic = json.load(infile)

	# set paths to datasets
	s_path = '../amazon/images'
	t1_path = '../dslr/images' 
	t2_path = '../webcam/images'

	# read in the datasets -> returns a dictionary in this format
	# {'source':{'x':X_s, 'y':Y_s}, 'target1':{'x':X_t1, 'y':Y_t1}, 'target2':{'x':X_t2, 'y':Y_t2}}
	datasets = read_datasets(s_path, t1_path, t2_path, label_dic)

	s_loader, s_size = create_dataloader(datasets['source'], data_transforms, batch_size)

	model = load_model("models/webcam.pt")

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model.to(device)

	logging.debug("loaded model trained on source domain")

	eval_model(model, s_loader['val'], device)





