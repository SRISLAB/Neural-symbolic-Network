# Functions of Always(G) and Eventurally(F)
import torch


MAX_LIMIT = 10.000 ** 6


# Calculate the robustness degree of G[tau1, tau2]miu
# output：The robustness degree of the STL after adding G
def Always(wrho):

    wrho = wrho.double()
    pos_wrho = torch.where(wrho >= 0.0, 1 + wrho, 1.0)
    pos_prod = torch.prod(pos_wrho, -1)
    pos_prod = pos_prod.double()
    pos_prod = torch.where(pos_prod < MAX_LIMIT, pos_prod, MAX_LIMIT * torch.sigmoid(pos_prod))
    pos_num = torch.where(torch.abs(wrho) > 0, 1.0, 0.0)
    power = torch.sum(pos_num, -1) + 0.000001
    pos_result = pos_prod ** (1 / power) - 1
    pos_result = pos_result.double()

    neg_whro = torch.where(wrho < 0.0, -wrho, 0.0) # torch.Size([20, 160, 255])
    neg_sum = torch.sum(-neg_whro, -1) # torch.Size([20, 160])
    neg_result = neg_sum / power

    result = torch.where(neg_result < 0, neg_result, pos_result)

    return result


# Calculate the robustness degree of F[tau1, tau2]miu
# output：The robustness degree of the STL after adding F
def Eventually(wrho):

    wrho = wrho.double()
    neg_wrho = torch.where(wrho <= 0.0, 1 - wrho, 1.0)
    neg_prod = torch.prod(neg_wrho, -1)
    neg_prod = neg_prod.double()
    neg_num = torch.where(torch.abs(wrho) > 0, 1.0, 0.0)
    neg_prod = torch.where(neg_prod < MAX_LIMIT, neg_prod, MAX_LIMIT * torch.sigmoid(neg_prod))
    power = torch.sum(neg_num, -1) + 0.000001
    neg_result = -neg_prod ** (1 / power) + 1

    pos_wrho = torch.where(wrho > 0, wrho, 0.0)
    pos_sum = torch.sum(pos_wrho, -1)
    pos_result = pos_sum / power

    result = torch.where(pos_result > 0, pos_result, neg_result)

    return result


