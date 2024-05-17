# Classes of modules in TLN (the most important py)
from Dependent_packages import Laplace
from Dependent_packages import Temporal_Operators
import torch.nn as nn
import torch
import torch.nn.functional as F


# Get the informative time intervals from input
def detect_time_range(input):

    B, N, L = input.size()
    time_range = torch.zeros(B, N, 2, dtype=torch.int64)

    for i in range(0, B):
        for j in range(0, N):
            nonzero_indices = torch.nonzero(input[i, j, :] != 0)
            if len(nonzero_indices) > 0:
                time_range[i, j, 0] = nonzero_indices[0]
                time_range[i, j, 1] = nonzero_indices[-1]

    return time_range


# Wavelet Convolution Module: features extraction
class Wavelet_Convolution(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Wavelet_Convolution, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        # Random generation of scale factors
        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):
        # Convolution with input signal
        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        p1 = time_disc - self.b_ / self.a_
        laplace_filter = Laplace.Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size)
        result = F.conv1d(waveforms, self.filters, stride=1, padding=16)
        return result


# Predicate Generation Module: calculate the robustness degree of predicates
# miu:= f(x[t]) ~ c, (c >= 0)
class Miu(nn.Module):

    # compare: >= 1, < -1
    # num_sig: the number of the features
    def __init__(self, compare, num_sig):
        super(Miu, self).__init__()
        self.compare = compare
        self.c = nn.Parameter(torch.randn(num_sig, 1), requires_grad=True)

    def forward(self, x):
        self.c.data = torch.clamp(self.c.data, min=0.0)
        if self.compare > 0:
            return x - self.c
        else:
            return self.c - x


# Temporal_A1: calculate the robustness degree of the STL which is concatenated by G
# phi:= miu1 G[tau1, tau2] miu2
class And_Convolution_2d(nn.Module):

    def __init__(self, kernal_size, stride, in_channels=1):
        super(And_Convolution_2d, self).__init__()
        self.And_2d_weight = nn.Parameter(abs(torch.randn(1, kernal_size)))
        self.stride = stride
        self.out_channels = 1
        self.in_channels = in_channels
        self.kernal_size = kernal_size
        nn.init.uniform_(self.And_2d_weight, 0.3, 0.7)

    def forward(self, input):
        TempLogic = Temporal_Operators.Always(input)
        torch.save(TempLogic, "./Model data/and_T.pt")
        N, L = TempLogic.size()

        current = None
        for i in range(0, L - self.kernal_size + 1, self.stride + 1):
            robustness = TempLogic[:, i:i + self.kernal_size]
            weight = torch.zeros(N, self.kernal_size, requires_grad=True) + self.And_2d_weight
            rho = torch.mul(robustness, weight)

            if current is not None:
                current = torch.cat((current, Temporal_Operators.Always(rho).view(N, -1)), 1)
            else:
                current = Temporal_Operators.Always(rho).view(N, -1)

        return current


# Temporal_O1: calculate the robustness degree of the STL which is concatenated by F
# phi:= miu1 F[tau1, tau2] miu2
class Or_Convolution_2d(nn.Module):

    def __init__(self, kernal_size, stride, in_channels=1):
        super(Or_Convolution_2d, self).__init__()
        self.Or_2d_weight = nn.Parameter(abs(torch.randn(1, kernal_size)))
        self.stride = stride
        self.out_channels = 1
        self.in_channels = in_channels
        self.kernal_size = kernal_size
        nn.init.uniform_(self.Or_2d_weight, 0.3, 0.7)

    def forward(self, input):
        TempLogic = Temporal_Operators.Eventually(input)
        torch.save(TempLogic, "./Model data/or_T.pt")
        N, L = TempLogic.size()

        current = None
        for i in range(0, L - self.kernal_size + 1, self.stride + 1):
            robustness = TempLogic[:, i:i + self.kernal_size]
            weight = torch.zeros(N, self.kernal_size, requires_grad=True) + self.Or_2d_weight
            rho = torch.mul(robustness, weight)
            if current is not None:
                current = torch.cat((current, Temporal_Operators.Eventually(rho).view(N, -1)), 1)
            else:
                current = Temporal_Operators.Eventually(rho).view(N, -1)

        return current


# Temporal_A2: calculate the robustness degree of the STL which is concatenated by G
# phi:= phi1 G[tau1, tau2] phi2
class And_Convolution(nn.Module):

    def __init__(self, kernal_size, stride, in_channels=1):
        super(And_Convolution, self).__init__()
        self.And_weight = nn.Parameter(abs(torch.randn(1, kernal_size)))
        self.stride = stride
        self.out_channels = 1
        self.in_channels = in_channels
        self.kernal_size = kernal_size
        nn.init.uniform_(self.And_weight, 0.3, 0.7)

    def forward(self, input):
        N, L = input.size()

        current = None
        for i in range(0, L - self.kernal_size + 1, self.stride):
            robustness = input[:, i:i + self.kernal_size]
            weight = torch.zeros(N, self.kernal_size, requires_grad=True) + self.And_weight
            rho = torch.mul(robustness, weight)

            if current is not None:
                current = torch.cat((current, Temporal_Operators.Always(rho).view(N, -1)), 1)
            else:
                current = Temporal_Operators.Always(rho).view(N, -1)

        return current


# Temporal_O2: calculate the robustness degree of the STL which is concatenated by F
# phi:= phi1 F[tau1, tau2] phi2
class Or_Convolution(nn.Module):

    def __init__(self, kernal_size, stride, in_channels=1):
        super(Or_Convolution, self).__init__()
        self.Or_weight = nn.Parameter(abs(torch.randn(1, kernal_size)))
        self.stride = stride
        self.out_channels = 1
        self.in_channels = in_channels
        self.kernal_size = kernal_size
        nn.init.uniform_(self.Or_weight, 0.3, 0.7)

    def forward(self, input):
        N, L = input.size()

        current = None
        for i in range(0, L - self.kernal_size + 1, self.stride):
            robustness = input[:, i:i + self.kernal_size]
            weight = torch.zeros(N, self.kernal_size, requires_grad=True) + self.Or_weight
            rho = torch.mul(robustness, weight)
            if current is not None:
                current = torch.cat((current, Temporal_Operators.Eventually(rho).view(N, -1)), 1)
            else:
                current = Temporal_Operators.Eventually(rho).view(N, -1)

        return current


# Temporal_U: Calculate the robustness degree of miu1 U[tau1, tau2] miu2
# miu:= miu1 U[tau1, tau2] miu2
def Until(input, until_time):

    # 1. 批量赋 0
    B, N, L = input.size()  # B = 20, N = 160, L = 127
    new_input = torch.zeros(until_time, B, N, L)  # [25,20,160,127]
    time_range = detect_time_range(input)  # [20,160,2]
    torch.save(time_range, './Model data/time_range.pt')

    mask = torch.zeros(B, N, L)  # [20,160,127]
    for b in range(B):
        for n in range(N):
            mask[b, n, int(time_range[b, n, 0]):int(time_range[b, n, 1] - until_time)] = 1.0

    for i in range(until_time):
        mask = torch.roll(mask, i, dims=2)  # [20,160,127]
        new_input[i] = torch.mul(input, mask)

    # 2. 在 new_input 的基础上添加 时序运算符
    group_1_1, group_1_2, \
    group_2_1, group_2_2, \
    group_3_1, group_3_2, \
    group_4_1, group_4_2 = torch.split(new_input, int(N / 8), dim=2)

    rho_group_1_1 = Temporal_Operators.Always(group_1_1)
    rho_group_1_2 = Temporal_Operators.Always(group_1_2)

    rho_group_2_1 = Temporal_Operators.Eventually(group_2_1)
    rho_group_2_2 = Temporal_Operators.Eventually(group_2_2)

    rho_group_3_1 = Temporal_Operators.Always(group_3_1)
    rho_group_3_2 = Temporal_Operators.Eventually(group_3_2)

    rho_group_4_1 = Temporal_Operators.Eventually(group_4_1)
    rho_group_4_2 = Temporal_Operators.Always(group_4_2)

    TempLogic_1 = torch.cat((rho_group_1_1,
                             rho_group_2_1,
                             rho_group_3_1,
                             rho_group_4_1), 2)  # 级联上述原子语句, [25,20,80]
    torch.save(TempLogic_1, "./Model data/until_T1.pt")

    TempLogic_2 = torch.cat((rho_group_1_2,
                             rho_group_2_2,
                             rho_group_3_2,
                             rho_group_4_2), 2)  # 级联上述原子语句, [25,20,80]
    torch.save(TempLogic_2, "./Model data/until_T2.pt")

    # 3. 计算 Until
    U, B, N = TempLogic_1.size()  # U = 25, B = 20, N = 80
    arr_2 = torch.zeros(B, N, U)  # [20, 80, 25]

    for t_i in range(until_time):
        rho_2 = TempLogic_2[t_i]
        arr_1 = torch.zeros(B, N, t_i + 1) # [20,80,t_i+1]
        for t_ii in range(t_i + 1):
            arr_1[:, :, t_ii] = TempLogic_1[t_ii]
        arr_1_min, index1 = torch.min(arr_1, dim=-1)
        rho_min = torch.where(rho_2 >= arr_1_min, arr_1_min, rho_2)
        arr_2[:, :, t_i] = rho_min
    ans, index2 = torch.max(arr_2, dim=-1)

    return ans






