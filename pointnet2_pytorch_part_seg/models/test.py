import torch


if __name__ == '__main__':
    B = 2
    S = 2
    group_idx = torch.arange(10).view(1, 1, 10).repeat([B, S, 1])
    print(group_idx)