import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

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


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class LossManager:
    def __init__(
            self,
            alpha,
            beta,
            gamma,
            initial_temperature,
            min_temperature,
            logit_stand
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.current_temperature = initial_temperature
        self.min_temperature = min_temperature


    def normalize(self, logit):
        mean = logit.mean(dim=-1, keepdims=True)
        stdv = logit.std(dim=-1, keepdims=True)

        return (logit - mean) / (1e-7 + stdv)

    def kd_loss(self, logits_student_in, logits_teacher_in, logit_stand=True):
        temperature = self.current_temperature

        logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
        logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
        log_pred_student = F.log_softmax(logits_student / temperature, dim=1)

        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss_kd *= temperature * temperature

        return loss_kd

    def perception(self, logits, epsilon=1e-5):
        """
        perform perception on logits.

        Parameters:
        logits (torch.Tensor): A tensor of shape (B, N) where B is the batch size and N is the number of classes.
        epsilon (float): A small constant to avoid division by zero in normalization.

        Returns:
        torch.Tensor: perception logits.
        """
        batch_mean = torch.mean(logits, dim=0, keepdim=True)
        batch_var = torch.var(logits, dim=0, keepdim=True, unbiased=False)
        x_normalized = (logits - batch_mean) / torch.sqrt(batch_var + epsilon)

        return x_normalized

    def luminet_loss(self, logits_student, logits_teacher, target):
        temperature = self.current_temperature
        stu_batch = self.perception(logits_student)
        tea_batch = self.perception(logits_teacher)

        pred_teacher = F.softmax(tea_batch / temperature, dim=1)
        log_pred_student = F.log_softmax(stu_batch / temperature, dim=1)

        nckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
        nckd_loss *= (33.0 * 33.0)

        return nckd_loss

    def cosine_loss(self, student_logits, teacher_logits):
        """
        Compute cosine similarity loss between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: Cosine similarity loss.
        """
        # Normalize logits
        student_norm = F.normalize(student_logits, p=2, dim=1)
        teacher_norm = F.normalize(teacher_logits, p=2, dim=1)

        # Compute cosine similarity loss
        cosine_loss = 1 - F.cosine_similarity(student_norm, teacher_norm).mean()
        return cosine_loss

    def rmse_loss(self, student_logits, teacher_logits):
        """
        Compute Root Mean Square Error (RMSE) between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: RMSE loss.
        """

        rmse = torch.sqrt(F.mse_loss(student_logits, teacher_logits))
        return rmse

    def mae_loss(self, student_logits, teacher_logits):
        """
        Compute Root Mean Square Error (RMSE) between student and teacher logits.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: RMSE loss.
        """

        rmse = torch.nn.L1Loss()(student_logits, teacher_logits)
        return rmse

    def hard_loss(self, student_logits, outputs):
        """
        Compute hard loss (cross-entropy) between student logits and true labels.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """

        return torch.nn.CrossEntropyLoss()(student_logits, outputs)

    def soft_distillation_loss(self, student_logits, teacher_logits):
        """
        Compute knowledge distillation loss with dynamic temperature.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.

        Returns:
            torch.Tensor: Knowledge distillation loss.
        """
        soft_targets = F.softmax(teacher_logits / self.current_temperature, dim=1)
        soft_predictions = F.log_softmax(student_logits / self.current_temperature, dim=1)

        loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        return loss * (self.current_temperature ** 2)

    def combined_loss(self, student_logits, teacher_logits, outputs):
        """Only include the additional losses (cosine and RMSE) here"""
        # Cosine loss
        cosine_loss = self.beta * self.cosine_loss(student_logits, teacher_logits)
        # RMSE loss
        rmse_loss = self.gamma * self.rmse_loss(student_logits, teacher_logits)
        return cosine_loss + rmse_loss

    def dkd_loss(self, logits_student_in, logits_teacher_in, target, alpha, beta, logit_stand=True):
        logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
        logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
        temperature = self.current_temperature

        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (temperature ** 2)
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
                * (temperature ** 2)
                / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss

class DynamicTemperatureScheduler(nn.Module):
    """
    Dynamic Temperature Scheduler for Knowledge Distillation.

    Args:
        initial_temperature (float): Starting temperature value.
        min_temperature (float): Minimum allowable temperature.
        max_temperature (float): Maximum allowable temperature.
        schedule_type (str): Type of temperature scheduling strategy.
        loss_type (str): Type of loss to use (combined or general KD).
        alpha (float): Importance for soft loss, 1-alpha for hard loss.
        beta (float): Importance of cosine loss.
        gamma (float): Importance for RMSE loss.
    """

    def __init__(
            self,
            initial_temperature=8.0,
            min_temperature=4.0,
            max_temperature=8,
            max_epoch=50,
            warmup=20,
            alpha=0.5,
            beta=0.9,
            gamma=0.5,
            logit_stand=True
    ):
        super(DynamicTemperatureScheduler, self).__init__()

        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_epoch = max_epoch
        self.warmup = warmup
        self.logit_stand = logit_stand
        # Constants for importance
        self.loss_manager = LossManager(
            alpha,
            beta,
            gamma,
            initial_temperature,
            min_temperature,
            logit_stand
        )


    def update_temperature(self, current_epoch, loss_divergence):
        progress = torch.tensor(current_epoch / self.max_epoch)
        cosine_factor = 0.5 * (1 + torch.cos(torch.pi * progress))
        log_loss = torch.log(torch.tensor(loss_divergence))
        adaptive_scale = log_loss / (log_loss + 1)

        if adaptive_scale > 1:
            target_temperature = self.initial_temperature * cosine_factor * (1 + adaptive_scale)
        else:
            target_temperature = self.initial_temperature * cosine_factor

        target_temperature = torch.clamp(
            target_temperature,
            self.min_temperature,
            self.max_temperature
        )

        momentum = 0.9
        self.current_temperature = momentum * self.current_temperature + (1 - momentum) * target_temperature

        self.loss_manager.current_temperature = self.current_temperature

    def get_temperature(self):
        """
        Retrieve current temperature value.

        Returns:
            float: Current dynamic temperature.
        """

        return self.current_temperature

    def forward(self, epoch, student_logits, teacher_logits, outputs, loss_type="kd"):
        """
        Forward pass to compute the loss based on the specified loss type.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        warmup = 1 if self.warmup is None else min(epoch / self.warmup, 1.0)

        if loss_type == "luminet":

            loss_ce = (2.0) * F.cross_entropy(
                student_logits,
                outputs
            )

            loss_luminet = warmup * self.loss_manager.luminet_loss(
                student_logits,
                teacher_logits,
                outputs
            )

            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_luminet,
            }

            return sum([l.mean() for l in losses_dict.values()])

        elif loss_type == "kd":
            logits_student = student_logits
            logits_teacher = teacher_logits

            target = outputs

            loss_ce = 0.1 * F.cross_entropy(logits_student, target)

            loss_kd = warmup * 9 * self.loss_manager.kd_loss(
                logits_student, logits_teacher, self.logit_stand
            )

            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }

            return sum([l.mean() for l in losses_dict.values()])

        elif loss_type == "dkd":
            logits_student = student_logits
            logits_teacher = teacher_logits

            target = outputs

            # losses
            loss_ce = 1.0 * F.cross_entropy(logits_student, target)

            loss_dkd = warmup * self.loss_manager.dkd_loss(
                logits_student,
                logits_teacher,
                target,
                1.0,
                8.0,
                self.logit_stand
            )

            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_dkd,
            }

            return sum([l.mean() for l in losses_dict.values()])
