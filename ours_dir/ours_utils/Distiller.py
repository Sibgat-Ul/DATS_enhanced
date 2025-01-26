import torch
import torch.nn as nn
import torch.nn.functional as F
from ours_dir.ours_utils.LossManager import LossManager

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
