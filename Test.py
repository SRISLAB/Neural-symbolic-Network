# Model Testing, saving data
from Dependent_packages import Dataload
from Dependent_packages import Modules
from Dependent_packages import Temporal_Grouping_Algorithm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


wavelet = 16
until_time = 25
current_directory = os.getcwd()
data_name = 'CWRU'
data_path = os.path.join(current_directory, data_name)
best_model_path = './Checkpoint/TLN_CWRU_1009-214757/14-0.7197-best_model.pth'
normalize_type = '0-1'
signal_size = 1024
input_channel = 1
output_channel = 10


# TLN for model test (little different from the original model)
class TLN(nn.Module):

    def __init__(self, in_channel=1, out_channel=10):
        super(TLN, self).__init__()

        # 1. Basic Predicate Network
        self.Basic_Predicate_Network = nn.Sequential(
            Modules.Wavelet_Convolution(wavelet, 32),
            nn.BatchNorm1d(wavelet),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.Miu_Pos = nn.Sequential(
            Modules.Miu(1, wavelet),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.Miu_Neg = nn.Sequential(
            Modules.Miu(-1, wavelet),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

        # 2. Auto-encoder
        self.AutoEncoder = torch.nn.Sequential(

            # 2.1 Encoder
            nn.Conv1d(2 * wavelet, 64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            # 2.2 Decoder
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(32, 64, kernel_size=3, padding=2),
            nn.Tanh(),

            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(64, 2 * wavelet, kernel_size=3, padding=2),
            nn.Sigmoid(),
            nn.MaxPool1d(kernel_size=4, stride=1)
        )

        # 3. Logic Network
        self.And_2d = nn.Sequential(
            Modules.And_Convolution_2d(20, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.Or_2d = nn.Sequential(
            Modules.Or_Convolution_2d(20, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.And = nn.Sequential(
            Modules.And_Convolution(6, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.Or = nn.Sequential(
            Modules.Or_Convolution(6, 2, 1),
            nn.ReLU(inplace=True)
        )

        # 4. Classifier
        self.Classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(170, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_channel)
        )

    def forward(self, x):

        x_feature = self.Basic_Predicate_Network(x)
        torch.save(x_feature, './Model data/x_feature.pt')

        x_pos = self.Miu_Pos(x_feature)
        x_neg = self.Miu_Neg(x_feature)

        x_miu = torch.cat((x_pos, x_neg), 1)
        torch.save(x_miu, './Model data/x_miu.pt')

        w = self.AutoEncoder(x_miu)
        torch.save(w, './Model data/w.pt')

        select = Temporal_Grouping_Algorithm.Segment(w)

        input = torch.tensor([])
        for k in range(0, select.size(0)):
            select_w = torch.zeros(w.size())
            select_w[select[k] == True] = w[select[k] == True]
            input = torch.cat((input, torch.mul(select_w, x_miu)), 1)

        torch.save(input, './Model data/input.pt')
        x_until_1 = Modules.Until(input, until_time)
        torch.save(x_until_1, './Model data/x_until_1.pt')

        x_and_1 = self.And_2d(input)
        torch.save(x_and_1, './Model data/x_and_1.pt')

        x_or_1 = self.Or_2d(input)
        torch.save(x_or_1, './Model data/x_or_1.pt')

        x_1 = torch.cat((x_and_1, x_or_1, x_until_1), 1)

        x_and_2 = self.And(x_1)
        torch.save(x_and_2, './Model data/x_and_2.pt')

        x_or_2 = self.Or(x_1)
        torch.save(x_or_2, './Model data/x_or_2.pt')

        x_2 = torch.cat((x_and_2, x_or_2), 1)

        ans = self.Classifier(x_2.float())

        return ans


# Load pre-train model
def load_model(best_model_path):

    model_object = TLN(input_channel, output_channel)
    pre_train_dict = torch.load(best_model_path)
    model_dict = model_object.state_dict()
    pre_train_dict = {k: v for k, v in pre_train_dict.items()
                     if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pre_train_dict)
    model_object.load_state_dict(model_dict)

    return model_object


# Load 20 testing data
def load_test_data():

    obj = Dataload.CWRU(data_path, normalize_type)
    train_dataset, val_dataset = obj.data_prepare(False)
    random_numbers = [random.randint(0, 1050) for _ in range(20)]
    train_tensor = torch.zeros(20, 1, signal_size)
    label_tensor = torch.zeros(20, 1)
    for i in random_numbers:
        label_tensor[random_numbers.index(i)] = torch.tensor(train_dataset[i][1])
        train_tensor[random_numbers.index(i)] = torch.tensor(train_dataset[i][0][0])
    torch.save(label_tensor, "./Model data/label_tensor.pt")
    torch.save(train_tensor, "./Model data/train_tensor.pt")

    return train_tensor, label_tensor


# Main func
if __name__ == '__main__':

    # train_tensor, label_tensor = load_test_data() # get new data
    train_tensor = torch.load('./Model data/train_tensor.pt')
    label_tensor = torch.load('./Model data/label_tensor.pt')
    model_object = load_model(best_model_path)
    output = model_object.forward(train_tensor) # get model data
    prob = F.softmax(output, dim=-1)
    N, L = prob.size()
    correct = 0
    with open('./Model data/test_result.txt', 'w') as f:
        for i in range(N):
            line = "Signal index: {}, " \
                   "original label: {}, " \
                   "predicate label: {}.\n".format(i, int(label_tensor[i]), int(prob[i].argmax(dim=-1)))
            f.write(line)
            if int(label_tensor[i]) == int(prob[i].argmax(dim=-1)):
                correct += 1
    accuracy = correct / N
    with open('./Model data/test_result.txt', 'a') as f:
        f.write("The accuracy of these {} test data is: {}.\n".format(N, accuracy))
