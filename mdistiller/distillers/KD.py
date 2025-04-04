import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.cfg = cfg

    def forward_train(self, image, target, **kwargs):
        if self.cfg.DATASET.TYPE != "tiny_imagenet":
            logits_student, _ = self.student(image)

            with torch.no_grad():
                logits_teacher, _ = self.teacher(image)
        else:
            logits_student = self.student(image)
            logits_teacher = self.teacher(image)
            print(target)
        # losses
        student_loss = F.cross_entropy(logits_student, target)
        teacher_loss = F.cross_entropy(logits_teacher, target)

        loss_ce = self.ce_loss_weight * student_loss
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }

        if self.cfg.SOLVER.TRAINER == "scheduler":
            loss_divergence = teacher_loss.item() - student_loss.item()
            return logits_student, losses_dict, loss_divergence

        return logits_student, losses_dict
