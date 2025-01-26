import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import transforms
from ours_utils.Distiller import DynamicTemperatureScheduler
from models import cifar_model_dict

dtkd_losses = []
dtkd_accuracies = []
our_losses = []
our_accuracies = []

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


def train_knowledge_distillation(
        name,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        lr,
        epochs=50,
        val_steps=10,
        temperature_scheduler=None,
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
        project="DTAD_Trials",
        name=name
    )

    # Optimizer and criterion
    student_optimizer = optimizer
    task_criterion = torch.nn.CrossEntropyLoss()

    # Set models to appropriate modes
    teacher_model.eval()
    val_loss = 0
    top1_acc = 0
    top5_acc = 0

    print("-" * 15 + " Teacher Validation " + "-" * 15)
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to("cuda", non_blocking=True), val_y.to("cuda", non_blocking=True)
            val_outputs = teacher_model(val_x)
            val_batch_loss = task_criterion(val_outputs, val_y)
            val_loss += val_batch_loss.item()

            # Calculate accuracies
            batch_top1, batch_top5 = calculate_accuracy(val_outputs, val_y)
            top1_acc += batch_top1.item()
            top5_acc += batch_top5.item()

    # Average validation metrics
    val_loss /= len(val_loader)
    top1_acc /= len(val_loader)
    top5_acc /= len(val_loader)

    print(f"Val Loss: {val_loss:.4f} | "
          f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%")
    print("-" * 50)

    best_top1_acc = 0.0  # Initialize best accuracy tracker

    # Training loop
    for epoch in range(epochs):
        # Training phase
        print("-" * 16 + " Training " + "-" * 16)
        student_model.train()
        train_loss = 0
        train_acc_1 = 0
        train_acc_5 = 0

        lr = adjust_learning_rate(epoch + 1, lr, student_optimizer) if scheduler == None else 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to('cuda', non_blocking=True), batch_y.to('cuda', non_blocking=True)

            # Forward passes
            with torch.no_grad():
                teacher_logits = teacher_model(batch_x)
                teacher_loss = task_criterion(teacher_logits, batch_y)

            student_logits = student_model(batch_x)
            student_loss = task_criterion(student_logits, batch_y)

            # Knowledge distillation loss
            if temperature_scheduler:
                # Combine losses
                total_batch_loss = temperature_scheduler(
                    epoch,
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    outputs=batch_y
                )

                temperature_scheduler.update_temperature(
                    current_epoch=epoch,
                    loss_divergence=teacher_loss.item() - student_loss.item()
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

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {total_batch_loss.item():.4f} | Temp: {temperature_scheduler.get_temperature():.2f} | "
                      f"Acc@1: {acc1.item():.2f}% | Acc@5: {acc5.item():.2f}%")

        # Epoch-end metrics
        train_loss /= len(train_loader)
        train_acc_1 /= len(train_loader)
        train_acc_5 /= len(train_loader)

        if scheduler is None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {train_loss:.4f} | "
              f"Acc@1: {train_acc_1:.2f}% | Acc@5: {train_acc_5:.2f}%")
        print("-" * 42)

        # if (epoch+1) % val_steps == 0:
        # Validation phase
        student_model.eval()
        val_loss = 0
        top1_acc = 0
        top5_acc = 0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to("cuda", non_blocking=True), val_y.to("cuda", non_blocking=True)
                val_outputs = student_model(val_x)
                val_batch_loss = task_criterion(val_outputs, val_y)
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
                "temp": temperature_scheduler.get_temperature()
            }
        )

        our_losses.append({"train_loss": train_loss, "test_loss": val_loss})
        our_accuracies.append({"acc@1": top1_acc, "acc@5": top5_acc})

        if (epoch + 1) % val_steps == 0:
            print("-" * 15 + " Validation " + "-" * 15)
            print(f"Epoch {epoch + 1}/{epochs} | Val Loss: {val_loss:.4f} | "
                  f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accuracy: {top5_acc:.2f}%")
            print("-" * 42)

        # Save the best model
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc

            if top1_acc > 60.0:
                torch.save(student_model.state_dict(), f"DTAD_@{top1_acc}.pth")
                print(f"Best model saved at epoch {epoch + 1} with Top-1 Accuracy: {best_top1_acc:.2f}%")
    print("Best Model Accuracy: ", best_top1_acc)
    run.finish()

    torch.save(student_model.state_dict(), "trained_studentDTAD.pth")
    return student_model

def main(teacher, student, key):
    # Define device
    teacher = teacher
    student = student
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.login(key=key)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    teacher_model, path = cifar_model_dict[teacher]
    teacher_model = teacher_model(num_classes=num_classes)
    teacher_model.load_state_dict(torch.load(path)["model"])
    teacher_model.to("cuda", non_blocking=True)

    print("model_loaded")

    student_model, path = cifar_model_dict[student]
    student_model = student_model(num_classes=num_classes)
    student_model.to("cuda", non_blocking=True)
    print("models loaded")

    max_epoch = 100
    lr = 0.05

    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=lr,
        weight_decay=5e-4,
        momentum=0.9
    )

    temp_scheduler = DynamicTemperatureScheduler(
        initial_temperature=4.0,
        min_temperature=2.0,
        max_temperature=8,
        max_epoch=max_epoch,
        warmup=20
    )

    trained_student = train_knowledge_distillation(
        f"{teacher}->{student} (Ours) + kd+norm4->2 + log scaling",
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        lr=lr,
        epochs=max_epoch,
        val_steps=20,
        temperature_scheduler=temp_scheduler,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ours_train")
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--teacher", type=str, default="resnet32x4")
    parser.add_argument("--student", type=str, default="resnet8x4")
    args = parser.parse_args()
    main(teacher=args.teacher, student=args.student, key=args.key)