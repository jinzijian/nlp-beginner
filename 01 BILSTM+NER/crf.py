import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

start_tag = -2
stop_tag = -1

class crf(nn.Moudle):
    def __init__(self, target_size, use_gpu):
        super(crf, self).__init__()
        self.use_gpu = use_gpu
        # 初始化 transition score
        self.target_size = target_size
        init_transitions = torch.zeros(self.target_size+2, target_size+2)
        init_transitions[:, start_tag] = -1000
        init_transitions[stop_tag, :] = -1000
        if self.use_gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)




