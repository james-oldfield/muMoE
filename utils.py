# mostly taken verbatim from: https://github.com/mlfoundations/wise-ft/blob/58b7a4b343b09dc06606aa929c2ef51accced8d1/src/models/utils.py

import os
import torch
import pickle
import json
import numpy as np
import math

def truncate(number, decimals):
    if decimals < 0:
        raise ValueError("Decimal places must be non-negative")
    elif decimals == 0:
        return math.floor(number)
    else:
        factor = 10 ** decimals
        return math.floor(number * factor) / factor

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_decay(start_value, end_value, num_steps):
    """
    Function to decay a scalar from start_value to end_value over num_steps using a cosine scheduler.
    
    Args:
    start_value (float): The starting value of the scalar.
    end_value (float): The ending value of the scalar.
    num_steps (int): The number of steps over which the decay should happen.

    Returns:
    torch.Tensor: A tensor containing the decayed values at each step.
    """
    steps = torch.arange(0, num_steps, dtype=torch.float32)
    cosine_decay_values = end_value + 0.5 * (start_value - end_value) * (1 + torch.cos(torch.pi * steps / num_steps))

    return cosine_decay_values

def linear_decay(start_value, end_value, num_steps):
    """
    Function to decay a scalar from start_value to end_value over num_steps using a linear scheduler.
    
    Args:
    start_value (float): The starting value of the scalar.
    end_value (float): The ending value of the scalar.
    num_steps (int): The number of steps over which the decay should happen.

    Returns:
    torch.Tensor: A tensor containing the decayed values at each step.
    """
    steps = torch.arange(0, num_steps, dtype=torch.float32)
    linear_decay_values = start_value + (end_value - start_value) * (steps / num_steps)

    return linear_decay_values

def exponential_decay(start_value, end_value, num_steps):
    """
    Function to decay a scalar from start_value to end_value over num_steps using an exponential scheduler.
    
    Args:
    start_value (float): The starting value of the scalar.
    end_value (float): The ending value of the scalar.
    num_steps (int): The number of steps over which the decay should happen.

    Returns:
    torch.Tensor: A tensor containing the decayed values at each step.
    """
    steps = torch.arange(0, num_steps, dtype=torch.float32)
    exponential_decay_values = start_value * (end_value / start_value) ** (steps / num_steps)

    return exponential_decay_values



def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

def linear_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = base_lr - (base_lr / es) * e
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_save(classifier, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier.cpu(), f)


def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def fisher_save(fisher, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fisher = {k: v.cpu() for k, v in fisher.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(fisher, f)


def fisher_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        fisher = pickle.load(f)
    if device is not None:
        fisher = {k: v.to(device) for k, v in fisher.items()}
    return fisher


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_IN_classes():
    # Load the class index to name mapping
    with open('/path_here/imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)

    idx_to_class = {int(k): v for k, v in class_idx.items()}
    return [idx_to_class[i][1] for i in range(len(idx_to_class))]

def get_caltech_classes(dataset_dir):
    # List all directories in the dataset root - each directory corresponds to a class
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    # Sort or process the class names as needed
    class_names = [x.split('.')[-1].split('-101')[0] for x in sorted(class_dirs)]
    return class_names
