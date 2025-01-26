import torch
import torch.nn as nn
import torch.nn.functional as F

class LossManager:
    def __init__(
            self,
            alpha,
            beta,
            gamma,
            initial_temperature,
            min_temperature
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

        logits_student = self.normalize(logits_student_in)
        logits_teacher = self.normalize(logits_teacher_in)
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