"""Utils for Datasets"""

import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader


class customDS(torch.utils.data.Dataset):
    def __init__(self, X,Y, transform=False):
        self.X = X
        self.Y = Y
        self.transform = transform
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x,y


def read_dataset(path, label_dic):
    """
    Given a path, returns arrays X and Y
    """
    labels = list(label_dic.keys())
    X = []
    Y = []
  
    # iterate through the class labels
    for label in labels:

        # path to label directory
        label_dir = os.path.join(path, label)

        # iterate through images in directory
        for img_file in os.listdir(label_dir):
            
            # append the label for this image
            Y.append(label_dic[label])
            # get the img path, read in the image, and append
            img_file_path = os.path.join(label_dir, img_file)
            img = np.array(Image.open(img_file_path))
            X.append(img)

    return np.array(X), np.array(Y)


def load_test_set(test_filenames, domain_dir_path, label_dic):
    """
    Given a list of test_filenames, form the 
    X, Y datasets for training and testing
    """
    X = []
    Y = []

    for filename in test_filenames:

        label, _ = filename.split("/")
        num_label = label_dic[label]
        Y.append(num_label)

        # get the img path, read in the image, and append
        img_file_path = os.path.join(domain_dir_path, filename)
        img = np.array(Image.open(img_file_path))
        X.append(img)

    trainX = []
    trainY = []
    labels = list(label_dic.keys())

    # iterate through the class labels
    for label in labels:

        # path to label directory
        label_dir = os.path.join(domain_dir_path, label)

        # iterate through images in directory
        for img_file in os.listdir(label_dir):

            if os.path.join(label, img_file) in test_filenames:
                continue
            
            # append the label for this image
            trainY.append(label_dic[label])
            # get the img path, read in the image, and append
            img_file_path = os.path.join(label_dir, img_file)
            img = np.array(Image.open(img_file_path))
            trainX.append(img)


    return {'x':np.array(trainX), 'y':np.array(trainY)}, {'x':np.array(X), 'y':np.array(Y)}



def class_count(Y):
    """
    Given the labels vector for a dataset, return
    a dictionary of label counts
    """
    count = {label: 0 for label in [i for i in range(0,31)]}
    for item in Y:
        count[item] += 1
    return count


def split_dataset(X, Y, samples_per_label, label_counts):
    """
    max_samples_per_label is the number of samples
    to use in training set
    """
    
    X_ref = []
    Y_ref = []
    X_test = []
    Y_test = []
    
    curr_idx = 0
    while curr_idx < len(Y):
        
        curr_label = Y[curr_idx]
        counts = label_counts[curr_label]
        
        idx_range = range(curr_idx, curr_idx + counts)
        size = counts if counts < samples_per_label else samples_per_label
        ref_sample = np.random.choice(range(curr_idx, curr_idx + size), size = size, replace=False)
        test_sample = np.asarray([x for x in idx_range if x not in ref_sample])
        
        if ref_sample != []:
            X_ref += list(X[ref_sample])
            Y_ref += [curr_label for _ in ref_sample]
        if test_sample != []:
            X_test += list(X[test_sample])
            Y_test += [curr_label for _ in test_sample]
        
        # increment counter
        curr_idx += counts

    return np.array(X_ref), np.array(Y_ref), np.array(X_test), np.array(Y_test)


def read_datasets(s_path, t1_path, t2_path,label_dic):
    X_s, Y_s = read_dataset(s_path,label_dic)
    X_t1, Y_t1 = read_dataset(t1_path,label_dic)
    X_t2, Y_t2 = read_dataset(t2_path,label_dic)
    return {'source':{'x':X_s, 'y':Y_s}, 'target1':{'x':X_t1, 'y':Y_t1}, 'target2':{'x':X_t2, 'y':Y_t2}}


def get_stream(X_ref, Y_ref, tg_X1, tg_Y1, tg_X2, tg_Y2, batch_size):
    """
    Return a generator over three domains
    """
    domain_idx = {0:[np.copy(X_ref), np.copy(Y_ref)],
                  1:[np.copy(tg_X1), np.copy(tg_Y1)],
                  2:[np.copy(tg_X2), np.copy(tg_Y2)]}

    while len(domain_idx) > 0:
    
        # randomly select a domain
        idx = np.random.choice(list(domain_idx.keys()))
        X, Y = domain_idx[idx]

        # abandon batches with fewer samples 
        if X.shape[0] < batch_size:
            print("Popped idx: %s" % idx)
            domain_idx.pop(idx)
            continue

        # randomly sample without replacement 
        sample_idx = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        yield X[sample_idx], Y[sample_idx], idx
    
        # remove returned samples from pool
        X = np.delete(X, sample_idx, axis=0)
        Y = np.delete(Y, sample_idx)

        if len(X) == 0:
            print("Popped idx: %s" % idx)
            domain_idx.pop(idx)
        else:
            domain_idx[idx] = [X, Y]



def create_dataloader(dataset, data_transforms, source = True, num_ref_per_class = 30, batch_size = 4):
    if source:
        counts = class_count(dataset['y'])
        X_ref, Y_ref, X_test, Y_test = split_dataset(dataset['x'], dataset['y'], num_ref_per_class, counts)
        dataset = {'train': customDS(X_ref, Y_ref, data_transforms['train']),  
                      'val': customDS(X_test, Y_test, data_transforms['test'])}
        loader = {x: DataLoader(dataset[x], batch_size, shuffle=True) for x in ['train', 'val']}
        dataset_size = {x: len(dataset[x]) for x in ['train', 'val']}
    else:
        dataset = customDS(dataset['x'], dataset['y'], data_transforms['test'])
        loader = {'val':DataLoader(dataset,batch_size=batch_size,shuffle=True)}
        dataset_size = {'val': len(dataset)}
    
    return loader, dataset_size


def mixed_dataloader(datasets, data_transforms, batch_size = 20, shuffle = False,num_ref_per_class = 30):
    count= class_count(datasets['source']['y'])
    X_ref, Y_ref, X_test, Y_test = split_dataset(datasets['source']['x'], datasets['source']['y'], num_ref_per_class, count)
    ref_dataset = customDS(X_ref, Y_ref, data_transforms['train'])
    ref_loader = DataLoader(ref_dataset,batch_size)
    ref_size = len(ref_dataset)
        
    dataset_0 = customDS(X_test, Y_test, data_transforms['test'])
    dataset_1 = customDS(datasets['target1']['x'], datasets['target1']['y'], data_transforms['test'])
    dataset_2 = customDS(datasets['target2']['x'], datasets['target2']['y'], data_transforms['test'])
    test_dataset = torch.utils.data.ConcatDataset([dataset_0, dataset_1,dataset_2])
    test_loader = DataLoader(test_dataset,shuffle=shuffle,batch_size=batch_size)
    return {'train': ref_loader, 'val':test_loader}, {'train': ref_size, 'val':len(dataset_0) + len(dataset_1) + len(dataset_2)}


