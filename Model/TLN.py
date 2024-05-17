# The structure of TLN
from Dependent_packages import Modules
from Dependent_packages import Temporal_Grouping_Algorithm
import torch
import torch.nn as nn


wavelet = 16
until_time = 35


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
        x_pos = self.Miu_Pos(x_feature)
        x_neg = self.Miu_Neg(x_feature)
        x_miu = torch.cat((x_pos, x_neg), 1)

        w = self.AutoEncoder(x_miu)
        select = Temporal_Grouping_Algorithm.Segment(w)

        input = torch.tensor([])
        for k in range(0, select.size(0)):
            select_w = torch.zeros(w.size())
            select_w[select[k] == True] = w[select[k] == True]
            input = torch.cat((input, torch.mul(select_w, x_miu)), 1)

        x_until_1 = Modules.Until(input, until_time)
        x_and_1 = self.And_2d(input)
        x_or_1 = self.Or_2d(input)
        x_1 = torch.cat((x_and_1, x_or_1, x_until_1), 1)

        x_and_2 = self.And(x_1)
        x_or_2 = self.Or(x_1)
        x_2 = torch.cat((x_and_2, x_or_2), 1)

        ans = self.Classifier(x_2.float())

        return ans
















