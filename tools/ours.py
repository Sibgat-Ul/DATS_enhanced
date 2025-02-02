import torch
import torch.nn as nn
import torch.nn.functional as F
from mdistiller.distillers import distiller_dict
import wandb
import numpy as np
from mdistiller.distillers.base import Distiller
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from mdistiller.models import cifar_model_dict
from mdistiller.engine.cfg import CFG as cfg

class DynamicTemperatureScheduler(nn.Module):
    def __init__(
            self,
            distiller: Distiller,
            initial_temperature=8.0,
            min_temperature=4.0,
            max_temperature=8,
            max_epoch=50,
            warmup=20,
    ):
        super(DynamicTemperatureScheduler, self).__init__()

        self.current_temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.max_epoch = max_epoch
        self.warmup = warmup

        # Constants for importance
        self.distiller = distiller

    def get_temperature(self):
        return self.distiller.temperature

    def update_temperature(self, current_epoch, loss_divergence):
        progress = torch.tensor(current_epoch / self.max_epoch)
        cosine_factor = 0.5 * (1 + torch.cos(torch.pi * progress))
        # log_loss = torch.log(torch.tensor(loss_divergence))
        adaptive_scale = loss_divergence / (loss_divergence + 1)

        if adaptive_scale > 1:
            target_temperature = self.initial_temperature * cosine_factor * (adaptive_scale)
        else:
            target_temperature = self.initial_temperature * cosine_factor

        target_temperature = torch.clamp(
            target_temperature,
            self.min_temperature,
            self.max_temperature
        )

        momentum = 0.9

        self.current_temperature = momentum * self.current_temperature + (1 - momentum) * target_temperature
        self.distiller.temperature = self.current_temperature

    def forward(self, epoch, image, target):
        preds, losses_dict, loss_divergence = self.distiller(image=image, target=target, epoch=epoch)
        return preds, sum([l.mean() for l in losses_dict.values()]), loss_divergence


def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """
    Calculate top-k accuracy

    Args:
        outputs (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth labels
        topk (tuple): Top-k values to compute accuracy

    Returns:
        list: Top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        # Get top-k predictions
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        # Calculate accuracies
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_learning_rate(epoch, lr, optimizer):
    steps = np.sum(epoch > np.asarray([62, 75, 87]))
    if steps > 0:
        new_lr = 0.05 * (0.1 ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return lr

our_loss = []
our_accuracy = []

def train_knowledge_distillation(
        name,
        train_loader,
        val_loader,
        optimizer,
        lr,
        epochs=50,
        val_steps=10,
        temperature_scheduler:DynamicTemperatureScheduler=None,
        scheduler=None,
        save_path="./output/"
):
    """
    Train student model with periodic validation

    Args:
        teacher_model (nn.Module): Pre-trained teacher model
        student_model (nn.Module): Model to be distilled
        train_dataset (Dataset): Training data
        val_dataset (Dataset): Validation data
        epochs (int): Total training epochs
        alpha (float): Loss balancing coefficient
        temperature_scheduler (DynamicTemperatureScheduler): Temperature scheduler
        save_path (str): Path to save the best model
    """

    run = wandb.init(
        # Set the project where this run will be logged
        project="DTS_Finale",
        name=name
    )

    # Optimizer and criterion
    student_optimizer = optimizer
    best_top1_acc = 0.0  # Initialize best accuracy tracker

    # Training loop
    for epoch in range(epochs):
        temperature_scheduler.distiller.train()
        # Training phase
        if (epoch + 1) % 100 == 0:
            print("-" * 16 + " Training " + "-" * 16)
        train_loss = 0
        train_acc_1 = 0
        train_acc_5 = 0

        lr = adjust_learning_rate(epoch + 1, lr, student_optimizer) if scheduler == None else 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to('cuda'), batch_y.to('cuda')

            # Combine losses
            student_logits, total_batch_loss, loss_divergence = temperature_scheduler(
                epoch=epoch,
                image=batch_x,
                target=batch_y
            )

            temperature_scheduler.update_temperature(
                current_epoch=epoch,
                loss_divergence=loss_divergence,
            )

            # Backward pass and optimization
            student_optimizer.zero_grad()
            total_batch_loss.backward()
            student_optimizer.step()

            # Calculate accuracies
            acc1, acc5 = calculate_accuracy(student_logits, batch_y)
            train_loss += total_batch_loss.item()
            train_acc_1 += acc1.item()
            train_acc_5 += acc5.item()

            if (epoch + 1) % 100 == 0:
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {total_batch_loss.item():.4f} | Temp: {temperature_scheduler.get_temperature():.2f} | "
                          f"Acc@1: {acc1.item():.2f}% | Acc@5: {acc5.item():.2f}%")

        # Epoch-end metrics
        train_loss /= len(train_loader)
        train_acc_1 /= len(train_loader)
        train_acc_5 /= len(train_loader)

        if scheduler != None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {train_loss:.4f} | "
                  f"Acc@1: {train_acc_1:.2f}% | Acc@5: {train_acc_5:.2f}%")
            print("-" * 42)

        temperature_scheduler.distiller.eval()
        val_loss = 0
        top1_acc = 0
        top5_acc = 0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to("cuda"), val_y.to("cuda")
                val_outputs, _ = student_model(val_x)
                val_batch_loss = F.cross_entropy(val_outputs, val_y)
                val_loss += val_batch_loss.item()

                # Calculate accuracies
                batch_top1, batch_top5 = calculate_accuracy(val_outputs, val_y)
                top1_acc += batch_top1.item()
                top5_acc += batch_top5.item()

        # Average validation metrics
        val_loss /= len(val_loader)
        top1_acc /= len(val_loader)
        top5_acc /= len(val_loader)

        wandb.log(
            {
                "train_acc": train_acc_1,
                "train_loss": train_loss,
                "val_acc": top1_acc,
                "val_loss": val_loss,
                "lr": lr,
                "temp": temperature_scheduler.distiller.temperature
            }
        )

        if (epoch + 1) % 10 == 0:
            print("-" * 15 + " Validation " + "-" * 15)
            print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | "
                  f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%")
            print("-" * 42)

        # Save the best model
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            run.summary["best_acc"] = best_top1_acc
            if top1_acc > 60.0 and (epoch + 1) > 87:
                torch.save(student_model.state_dict(), f"./output/DTS_best.pth")
                print(f"Best model saved at epoch {epoch + 1} with Top-1 Accuracy: {best_top1_acc:.2f}%")

    print("Best Model Accuracy: ", best_top1_acc)
    run.finish()

    return student_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")

    parser.add_argument("--exp_name", type=str, help="experiment name", default="DTAD_Experiment")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--distiller_type", type=str, default="NONE")

    parser.add_argument("--resume", type=bool, default=False)

    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--init_temperature", type=float, default=4.0)
    parser.add_argument("--min_temperature", type=float, default=2.0)
    parser.add_argument("--max_temperature", type=float, default=4.0)
    parser.add_argument("--adjust_temperature", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--logit_stand", action="store_true")
    parser.add_argument("--base_temp", type=float, default=2)
    parser.add_argument("--kd_weight", type=float, default=9)
    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    cfg.SOLVER.TRAINER = "scheduler"
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.LOG.WANDB = True

    cfg.freeze()

    teacher = cfg.DISTILLER.TEACHER
    student = cfg.DISTILLER.STUDENT
    loss_type = cfg.DISTILLER.TYPE
    logit_stand = args.logit_stand
    min_temp = 2 if logit_stand else 4
    max_temp = min_temp * 2

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 Data Preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Mean and std of CIFAR-10
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, transform=transform, download=True)
    val_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, transform=val_transform, download=True)
    num_classes = len(train_dataset.classes)

    batch_size = 128
    max_epoch = 100
    lr = 0.05

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    teacher_model, path = cifar_model_dict[teacher]
    teacher_model = teacher_model(num_classes=num_classes)
    teacher_model.load_state_dict(torch.load(path)["model"])
    teacher_model.to("cuda")

    print("model_loaded")

    student_model = cifar_model_dict[student][0](
            num_classes=num_classes
    )

    student_model.to("cuda")
    print("models loaded")

    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        weight_decay=5e-4,
        momentum=0.9
    )

    temp_scheduler = DynamicTemperatureScheduler(
        distiller_dict[cfg.DISTILLER.TYPE](student_model, teacher_model, cfg),
        initial_temperature=max_temp,
        min_temperature=min_temp,
        max_temperature=max_temp,
        max_epoch=max_epoch,
        warmup=20
    )

    temp_scheduler.loss_type = cfg.DISTILLER.TYPE
    temp_scheduler.logit_stand = logit_stand

    # Set models to appropriate modes
    teacher_model.eval()
    val_loss = 0
    top1_acc = 0
    top5_acc = 0

    # print("-" * 15 + " Teacher Validation " + "-" * 15)
    # with torch.no_grad():
    #     for val_x, val_y in val_loader:
    #         val_x, val_y = val_x.to("cuda"), val_y.to("cuda")
    #         val_outputs, _ = teacher_model(val_x)
    #         val_batch_loss = F.cross_entropy(val_outputs, val_y)
    #         val_loss += val_batch_loss.item()
    #
    #         # Calculate accuracies
    #         batch_top1, batch_top5 = calculate_accuracy(val_outputs, val_y)
    #         top1_acc += batch_top1.item()
    #         top5_acc += batch_top5.item()
    #
    # # Average validation metrics
    # val_loss /= len(val_loader)
    # top1_acc /= len(val_loader)
    # top5_acc /= len(val_loader)
    #
    # print(f"Val Loss: {val_loss:.4f} | "
    #       f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%")
    # print("-" * 50)

    exp_name = f"{teacher}->{student}(Ours) {loss_type} + {logit_stand} {max_temp}->{min_temp}"
    trained_student = train_knowledge_distillation(
        name=exp_name,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr=lr,
        epochs=max_epoch,
        val_steps=20,
        temperature_scheduler=temp_scheduler
    )