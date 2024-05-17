# Temporal Grouping Algorithm
import torch


# scores: probability sequence
# max_p: the threshold of the maximum probability
# pro_p: the threshold of the maximum range
# select: mask with the continuous informative range
def Segment(scores):

    eta = 5.0
    B, C, W = scores.size()  # [20, 32, 255]
    min_value, idxm = torch.min(scores, dim=-1)
    norm = torch.matmul(min_value.view(scores.size(0), scores.size(1), -1), torch.ones(1, scores.size(2)))
    n_scores = scores - norm  # make the minimal value of each tensor 0

    value, indix = torch.max(n_scores, dim=-1)
    value = torch.matmul(value.view(scores.size(0), scores.size(1), -1), torch.ones(1, scores.size(2)))

    result_metrics = torch.tensor([])
    scores_2 = torch.mul(scores, scores)  # the square of probability sequence
    mean_score = torch.zeros(scores.size())
    mean_scores = torch.zeros(scores.size())

    for m in range(W):
        mean_score[:, :, m] = torch.sum(scores[:, :, 0:m + 1], -1)
        mean_scores[:, :, m] = torch.sum(scores_2[:, :, 0:m + 1], -1)

    for delta in range(2, 10):
        boolean_scores = torch.gt(n_scores, delta * value / 10.0)
        false_extend = torch.zeros(B, C, 1, dtype=torch.bool)

        shift_right = torch.cat((boolean_scores, false_extend), 2)
        shift_left = torch.cat((false_extend, boolean_scores), 2)
        idx_bool = torch.logical_xor(shift_left, shift_right)
        idx_boole = torch.logical_and(idx_bool, shift_left)
        idx_bools = torch.logical_and(idx_bool, shift_right)

        nonzero_start = torch.nonzero(idx_bools)
        nonzero_end = torch.nonzero(idx_boole)

        duration = nonzero_end - nonzero_start

        new_start = torch.cat((nonzero_start, duration), 1)

        sum_duration = mean_score[new_start[:, 0], new_start[:, 1], new_start[:, 2] + new_start[:, 5] - 1] - \
                       mean_score[new_start[:, 0], new_start[:, 1], new_start[:, 2]]
        mean_duration = torch.div(sum_duration, new_start[:, 5])
        sum_durations = mean_scores[new_start[:, 0], new_start[:, 1], new_start[:, 2] + new_start[:, 5] - 1] - \
                        mean_scores[new_start[:, 0], new_start[:, 1], new_start[:, 2]]
        mean_durations = torch.div(sum_durations, new_start[:, 5])
        var_duration = torch.sqrt(mean_durations - mean_duration ** 2)

        new_start[:, 3] = 100 * (
                0.01 * torch.sqrt(torch.tensor(delta)) * sum_duration + mean_duration + eta * var_duration)
        new_start[:, 4] = torch.max(new_start[:, 2] - (eta * var_duration).int(), torch.zeros(new_start.size(0)))
        new_start[:, 5] = torch.min(W * torch.ones(new_start.size(0)), new_start[:, 5] + (2 * eta * var_duration).int())

        result_metrics = torch.cat((result_metrics, new_start), 0)

    K = 5
    select = torch.zeros(K, B, C, W, dtype=torch.bool)

    for i in range(B):
        for j in range(C):
            start_ij = result_metrics[(result_metrics[:, 0] == i) & (result_metrics[:, 1] == j), :].int()
            metric = start_ij[:, 3]
            max_value, sort_idx = torch.sort(metric, descending=True)

            if sort_idx.size(0) >= K:
                for k in range(0, K):
                    first = sort_idx[k]
                    select[k, i, j, start_ij[first, 4]:start_ij[first, 4] + start_ij[first, 5]] = True

    return select



