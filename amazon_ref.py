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
	Run streaming simulation on all batch sizes

	If we are loading a trained model, we need to read in the test set,
	which will be used for the stream
	"""
	
	# load label dictionary
	with open("models/label_dict.json", "r") as infile:
		label_dic = json.load(infile)

	# set paths to datasets
	# amazon is the source/reference domain
	s_path = '../amazon/images'
	t1_path = '../dslr/images' 
	t2_path = '../webcam/images'

	# read in the datasets -> returns a dictionary in this format
	# {'source':{'x':X_s, 'y':Y_s}, 'target1':{'x':X_t1, 'y':Y_t1}, 'target2':{'x':X_t2, 'y':Y_t2}}
	# we only need the data at target1 and target2
	datasets = read_datasets(s_path, t1_path, t2_path, label_dic)

	logging.info("loaded dslr dataset, size: %s", datasets["target1"]['x'].shape[0])
	logging.info("loaded webcam dataset, size: %s", datasets["target2"]["x"].shape[0])

	# given the test image filenames, read in the test data
	test_filenames = np.load("models/amazon_test_filenames.npy")
	ref_train, ref_test = load_test_set(test_filenames, s_path, label_dic)

	logging.info("loaded training set of size: %s", ref_train['x'].shape[0])
	logging.info("loaded test reference set of size: %s", ref_test['x'].shape[0])

	# load trained model on reference domain
	model = load_model("models/amazon_new.pt")
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

	time_per_batch = []

	for batch_size in batch_sizes:

		logging.info("starting simulation for batch size %s", batch_size)

		domain_ks = {0:[], 1:[], 2:[]}

		# create stream
		drift_bool = {0:False, 1:True, 2:True}
		stream = get_stream(ref_test['x'], ref_test['y'], datasets["target1"]['x'], 
			                datasets["target1"]['y'], datasets["target2"]['x'],
			                datasets["target2"]['y'], drift_bool, batch_size)
		
		logging.info("created stream")

		batch_time = 0
		accuracy = []
		drifts = []
		for batch_X, batch_Y, domain_idx in stream:


			batch_set = customDS(batch_X, batch_Y, data_transforms['test'])
			batch_loader = DataLoader(batch_set, batch_size=batch_size, shuffle=True)

			# accs and drift_pos are lists containing one number
			accs, drift_pos, times, start, ks_stats = drift_statistics(batch_loader, model, drift_detector, device)

			batch_time += times[-1] - start
			accuracy += accs
			drifts += drift_pos

			domain_ks[domain_idx] += ks_stats

		# print p-value statistics
		for idx in range(0, 3):
			logging.info("statistics for domain %s : min/max/avg/std - %s/%s/%s/%s", 
				          idx, min(domain_ks[idx]), max(domain_ks[idx]), 
				          sum(domain_ks[idx])/len(domain_ks[idx]), 
				          np.std(np.array(domain_ks[idx])))

		time_per_batch.append(batch_time)

		t_p, f_p, f_n, t_n = confusion_matrix(accuracy, drifts, initial_accuracy - 0.10)

		logging.info("Confusion Matrix: TP/FP/FN/TN - %s/%s/%s/%s", t_p, f_p, f_n, t_n)

	logging.info("Time per batch: %s", time_per_batch)



if __name__ == '__main__':

	run()
