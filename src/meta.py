import torch
import numpy as np
from torch import nn
from torch import optim
from learner import Learner


def deviation_loss_torch(y_true, y_pred):
    assert y_true.detach().cpu().shape == y_pred.detach().cpu().shape
    confidence_margin = 5
    device = y_pred.device
    data_num = y_pred.size()[0]
    ref = np.random.normal(loc=0., scale=1.0, size=5000)
    ref_torch = torch.from_numpy(ref).to(device)
    dev = (y_pred - torch.mean(ref_torch)) / torch.std(ref_torch)
    inlier_loss = torch.abs(dev)
    zero_array = torch.from_numpy(np.zeros((data_num, 1))).to(device)
    outlier_loss = torch.abs(torch.maximum(confidence_margin - dev, zero_array))
    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


class Meta_episodic(nn.Module):
    def __init__(self, args, config):
        super(Meta_episodic, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.totalTask_num = args.totalTask_num
        self.update_step = args.update_step
        self.net = Learner(config)
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr, weight_decay=args.weight_decay)

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        totalTask_num = self.totalTask_num
        losses_q = [0 for _ in range(self.update_step + 1)]
        auged_task_num = len(x_spt)
        for i in range(auged_task_num):
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = deviation_loss_torch(y_spt[i], logits)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = deviation_loss_torch(y_qry[i], logits_q)
                losses_q[0] += loss_q
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = deviation_loss_torch(y_qry[i], logits_q)
                losses_q[1] += loss_q
            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = deviation_loss_torch(y_spt[i], logits)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = deviation_loss_torch(y_qry[i], logits_q)
                losses_q[k + 1] += loss_q
        loss_q = losses_q[-1] / auged_task_num
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        loss_value = loss_q.item()
        return loss_value

    def testing_GAD(self, x_test):
        final_pred_score = self.net(x_test, vars=None, bn_training=False)
        return final_pred_score
