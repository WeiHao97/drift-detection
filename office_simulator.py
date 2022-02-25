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
import sys

warnings.filterwarnings('ignore')

from utils.drift_detection import *
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
	for inputs, labels in s_loader:
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


def run():
	"""
	Run streaming simulation on all batch sizes without online training

	If we are loading a trained model, we need to read in the test set,
	which will be used for the stream
	"""
	
	# load label dictionary
	with open("models/label_dict.json", "r") as infile:
		label_dic = json.load(infile)

	# set paths to datasets
	s_path = sys.argv[1] # '../amazon/images'
	t1_path = sys.argv[2] # '../dslr/images' 
	t2_path = sys.argv[3] # '../webcam/images'

	# read in the datasets -> returns a dictionary in this format
	# {'source':{'x':X_s, 'y':Y_s}, 'target1':{'x':X_t1, 'y':Y_t1}, 'target2':{'x':X_t2, 'y':Y_t2}}
	# we only need the data at target1 and target2
	datasets = read_datasets(s_path, t1_path, t2_path, label_dic)

	logging.info("loaded first target dataset, size: %s", datasets["target1"]['x'].shape[0])
	logging.info("loaded second target dataset, size: %s", datasets["target2"]["x"].shape[0])

	# given the test image filenames, read in the test data
	test_filenames = np.load(sys.argv[4]) # "models/amazon_test_filenames.npy"
	ref_train, ref_test = load_test_set(test_filenames, s_path, label_dic)

	logging.info("loaded ref training set of size: %s", ref_train['x'].shape[0])
	logging.info("loaded ref test reference set of size: %s", ref_test['x'].shape[0])

	# load trained model on reference domain
	model = load_model(sys.argv[5]) # "models/amazon_new.pt"
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	logging.debug("loaded model trained on source domain")

	# create data loader for training set and construct drift detector
	dataset = customDS(ref_train['x'], ref_train['y'], data_transforms['train'])
	loader = DataLoader(dataset, batch_size=20, shuffle=True)
	logits = compute_logits(model, device, loader) # get logits
	drift_detector = drift_detection(logits, 0.05, method="KSDrift")

	test_dataset = customDS(ref_test['x'], ref_test['y'], data_transforms['test'])
	test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)

	initial_accuracy = eval_model(model, test_loader, device).item()
	logging.info("initial accuracy of trained model: %s", initial_accuracy)


	##########################
	# SIMULATIONS START HERE #
	##########################

	if not os.path.isdir("./out"):
		os.mkdir("./out")

	time_per_batch = {}
	confusion = {batch_size:{"fpr":[], "tpr":[]} for batch_size in batch_sizes}

	for batch_size in batch_sizes:

		logging.info("starting simulation for batch size %s", batch_size)

		# create stream
		stream = get_stream(ref_test['x'], ref_test['y'], datasets["target1"]['x'], 
			                datasets["target1"]['y'], datasets["target2"]['x'],
			                datasets["target2"]['y'], batch_size)
		
		logging.info("created stream")

		batch_time = 0
		accuracy = {0:[], 1:[], 2:[]}
		drifts = {0:[], 1:[], 2:[]}
		entropies = {0:[], 1:[], 2:[]}
		for batch_X, batch_Y, domain_idx in stream:


			batch_set = customDS(batch_X, batch_Y, data_transforms['test'])
			batch_loader = DataLoader(batch_set, batch_size=batch_size, shuffle=True)

			# accs and drift_pos are lists containing one number
			accs, drift_pos, uncertainties, times, start = drift_statistics(batch_loader, model, drift_detector, device)

			batch_time += times[-1] - start
			accuracy[domain_idx].append(accs[0].item())
			drifts[domain_idx].append(drift_pos[0])
			entropies[domain_idx].append(uncertainties[0])
			

		# append time to process batch
		time_per_batch[batch_size] = batch_time


		# get false positive rate and true positive rate
		fpr, tpr = confusion_matrix(accuracy, drifts, initial_accuracy - 0.10)
		confusion[batch_size]["fpr"].append(fpr)
		confusion[batch_size]["tpr"].append(tpr)

		logging.info("Confusion Matrix: FPR/TPR - %s/%s", fpr, tpr)

		# save batch data
		with open("./out/accuracy_%s.json" % batch_size, "w") as file:
			json.dump(accuracy, file)
		with open("./out/drifts_%s.json" % batch_size, "w") as file:
			json.dump(drifts, file)
		with open("./out/entropies_%s.json" % batch_size, "w") as file:
			json.dump(entropies, file)


	# save evalution time and TPR/FPR 
	with open("./out/time_per_batch.json", "w") as file:
		json.dump(time_per_batch, file)
	with open("./out/confusion.json", "w") as file:
		json.dump(confusion, file)


	logging.info("Time per batch: %s", time_per_batch)


if __name__ == '__main__':

	run()




