import random
import numpy as np
import torch
from utils.dataloader import get_loader
import os
import codecs
import csv
from copy import deepcopy
# from utils.model import ImageNet, AudioNet
from utils.model_res import ImageNet, AudioNet
from utils.dist_utils import *
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from geomloss import SamplesLoss
import geomloss
from utils.AVEDataset import AVEDataset
# from utils.VGGSoundDataset import VGGSound
from torch.utils.data import DataLoader
import math
from utils.shake import Shake
import wandb
from scipy.stats.stats import kendalltau

epoch_for_tag = 50
epoch_for_retrain = 50
# learning_rate = 0.005
# momentum = 0.9
# place_image = 4
# place_audio = 4

repeat_permute = 1
max_permute_inner = 200

import torch.nn as nn


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    # aaa = F.cosine_similarity(y_s, y_t, dim=-1)
    # return F.cosine_similarity(y_s, y_t, dim=-1).mean()
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        # kd_loss = self.gamma * intra_loss
        return kd_loss


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

class DisLoss(torch.nn.Module):
    def __init__(self, type):
        super(DisLoss, self).__init__()
        self.type = type

    def forward(self, x, y):
        if self.type == 1:
            return torch.norm(x - y, p=2) / x.shape[0] / x.shape[1]


def dkd_loss(logits_student, logits_teacher, target, alpha = 1, beta = 8, temperature = 4):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def ntkl(logits_student, logits_teacher, target, mask=None, criterion4=None, temperature=1):

    gt_mask = _get_gt_mask(logits_student, target)
    logits_teacher = logits_teacher * (~gt_mask)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature, dim=1)
    logits_student = logits_student * (~gt_mask)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature, dim=1)

    if mask.sum() == 0:
        temp = torch.tensor(0)
    else:
        # temp = ((mask * (criterion4(log_pred_student_part2, pred_teacher_part2).sum(1))).sum())/mask.sum()
        # temp = ((weight * mask * (- log_pred_student_part2 * pred_teacher_part2).sum(1)).sum(0))/mask.sum()
        temp = ((mask * (criterion4(log_pred_student_part2, pred_teacher_part2.detach()).sum(1)))).mean()
    return temp

