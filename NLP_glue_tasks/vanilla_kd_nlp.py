import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DistillationPipeline:
    def __init__(self, teacher_checkpoint, student_checkpoint, num_labels, device='cuda'):
        # Initialize models
        self.teacher = AutoModelForSequenceClassification.from_pretrained(teacher_checkpoint, num_labels=num_labels)
        self.student = AutoModelForSequenceClassification.from_pretrained(student_checkpoint, num_labels=num_labels)

        # Initialize tokenizers
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_checkpoint)
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_checkpoint)

        self.device = device
        self.teacher.to(device).eval()
        self.student.to(device)

    def compute_loss(self, student_logits, teacher_logits, labels=None, alpha=0.5, temperature=2.0):
        # Soften logits with temperature
        soft_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        # KL Divergence Loss
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

        # Optional: Combine with Cross-Entropy
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels.to(self.device))
            return alpha * kl_loss + (1. - alpha) * ce_loss
        return kl_loss

    def distill(self, texts, optimizer, alpha=0.5, temperature=2.0):
        # Tokenize separately for each model
        teacher_inputs = self.teacher_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(
            self.device)
        student_inputs = self.student_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(
            self.device)

        # Teacher Forward (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**teacher_inputs).logits

        # Student Forward
        student_outputs = self.student(**student_inputs).logits

        # Compute Loss
        loss = self.compute_loss(student_outputs, teacher_outputs, None, alpha, temperature)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()