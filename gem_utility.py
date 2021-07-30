import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog

def compute_offset(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

def sotre_grad(pp, grad, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters 参数
        grad: gradients 梯度
        grad_dims: list with number of parameters per layers 每层的参数数量
        tid: task id 任务id
    """
    # store the gradients
    grad[:,tid].fill_(0,0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grad[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp,newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed gradient "gradient", 
    """