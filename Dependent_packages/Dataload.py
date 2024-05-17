# Functions related to data loading
import os
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm
from Dependent_packages.Preprocess import *
from Dependent_packages.Sequence import dataset

signal_size = 1024
dataset_name = ["Normal Baseline Data", "12k Drive End Bearing Fault Data"]

# 12kHz, 0hp, 1797rpm, Drive End
normal_signal = "97.mat"
fault_signal = ["105.mat", "169.mat", "209.mat",  # inner: 0.007 inches, 0.014 inches, 0.021 inches
                "130.mat", "197.mat", "234.mat",  # outer: 0.007 inches, 0.014 inches, 0.021 inches
                "118.mat", "185.mat", "222.mat"]  # ball: 0.007 inches, 0.014 inches, 0.021 inches

# The normal data is labeled 0, The failure data is labeled 1-9
label = [1, 2, 3, 4, 5, 6, 7, 8, 9]
axis = ["_DE_time", "_FE_time", "_BA_time"]


# Get files
def get_files(root, test=False):

    data_root1 = os.path.join('/tmp', root, dataset_name[0])
    data_root2 = os.path.join('/tmp', root, dataset_name[1])

    path1 = os.path.join('/tmp', data_root1, normal_signal)
    data, lab = data_load(path1, axis_name=normal_signal, label=0)

    for i in tqdm(range(len(fault_signal))):
        path2 = os.path.join('/tmp', data_root2, fault_signal[i])
        data1, lab1 = data_load(path2, axis_name=fault_signal[i], label=label[i])
        data += data1
        lab += lab1
    return [data, lab]


# Generate Training Dataset and Testing Dataset
def data_load(filename, axis_name, label):

    data_number = axis_name.split(".")
    if eval(data_number[0]) < 100:
        real_axis = "X0" + data_number[0] + axis[0]
    else:
        real_axis = "X" + data_number[0] + axis[0]
    fl = loadmat(filename)[real_axis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
    return data, lab


# Data preprocess
def data_transforms(dataset_type, normalize_type):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normalize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            RandomCrop(),
            Retype()
        ]),

        'val': Compose([
            Reshape(),
            Normalize(normalize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


# Train / test data split
def train_test_split_order(data_pd, test_size, num_classes):

    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], True)
    return train_pd, val_pd


# The class of CWRU dataset
class CWRU(object):

    input_channel = 1
    num_classes = 10

    def __init__(self, data_dir, normalize_type):

        self.data_dir = data_dir
        self.normalize_type = normalize_type

    def data_prepare(self, test=False):

        list_data = get_files(self.data_dir, test)
        if test:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            test_dataset = dataset(list_data=data_pd, test=True, transform=None)
            return test_dataset
        else:
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split_order(data_pd, test_size=0.2, num_classes=10)
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normalize_type))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normalize_type))
            return train_dataset, val_dataset



