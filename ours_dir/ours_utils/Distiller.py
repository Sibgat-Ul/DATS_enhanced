import torch
import torch.nn as nn
import torch.nn.functional as F
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
    ):
        super(DynamicTemperatureScheduler, self).__init__()

        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_epoch = max_epoch
        self.warmup = warmup

        # Tracking training dynamics
        self.loss_history = []
        self.student_loss = []

        # Constants for importance
        self.loss_manager = LossManager(
            alpha,
            beta,
            gamma,
            initial_temperature,
            min_temperature
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

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

    def forward(self, epoch, student_logits, teacher_logits, outputs, loss_type="kd++"):
        """
        Forward pass to compute the loss based on the specified loss type.

        Args:
            student_logits (torch.Tensor): Logits from student model.
            teacher_logits (torch.Tensor): Logits from teacher model.
            outputs (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        if loss_type == "ours":
            temp_ratio = (self.current_temperature - 1.0) / (3.0 - 1.0)
            temp_ratio = max(0, min(1, temp_ratio))

            # Base losses (always present)
            soft_loss = self.loss_manager.soft_distillation_loss(
                student_logits,
                teacher_logits
            )

            hard_loss = self.loss_manager.hard_loss(
                student_logits,
                outputs
            )

            teacher_loss = self.loss_manager.hard_loss(
                teacher_logits,
                outputs
            )

            # Temperature-dependent weighting for soft vs hard
            if self.current_temperature > 1:
                soft_weight = self.alpha * temp_ratio + 0.4 * (1 - temp_ratio)
                hard_weight = (1 - self.alpha) * temp_ratio + 0.5 * (1 - temp_ratio)
            else:
                soft_weight = 0.2
                hard_weight = 0.5

            # Additional losses only when temperature is higher
            additional_losses = temp_ratio * self.loss_manager.combined_loss(
                student_logits,
                teacher_logits,
                outputs
            )

            warmup = 1 if self.warmup == None else min(epoch / self.warmup, 1.0)

            total_loss = (
                    soft_weight * soft_loss +
                    hard_weight * hard_loss +
                    additional_losses
            )

            return warmup * total_loss

        elif loss_type == "luminet":
            warmup = 1 if self.warmup == None else min(epoch / self.warmup, 1.0)

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

        elif loss_type == "kd++":
            logits_student = student_logits
            logits_teacher = teacher_logits

            target = outputs

            loss_ce = 0.1 * F.cross_entropy(logits_student, target)

            loss_kd = 9 * self.loss_manager.kd_loss(
                logits_student, logits_teacher
            )

            losses_dict = {
                "loss_ce": loss_ce,
                "loss_kd": loss_kd,
            }

            return sum([l.mean() for l in losses_dict.values()])
