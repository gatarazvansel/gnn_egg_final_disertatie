from __future__ import annotations

import os
import copy
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models.model_factory import get_model, normalize_model_name
from src.dataset_builder import get_class_configuration, normalize_classification_mode


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset_path: str) -> list:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please build the dataset first."
        )

    dataset = torch.load(dataset_path, weights_only=False)

    if not isinstance(dataset, list) or len(dataset) == 0:
        raise ValueError("Loaded dataset is empty or invalid.")

    return dataset


def validate_dataset_metadata(dataset: list) -> None:
    missing_subject_id = [
        idx for idx, graph in enumerate(dataset)
        if not hasattr(graph, "subject_id")
    ]
    if missing_subject_id:
        raise ValueError(
            "Some graphs are missing 'subject_id'. Rebuild the dataset using the new dataset builder."
        )

    missing_labels = [
        idx for idx, graph in enumerate(dataset)
        if not hasattr(graph, "y")
    ]
    if missing_labels:
        raise ValueError("Some graphs are missing labels 'y'.")


def split_dataset_by_subject(
    dataset: list,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    subject_to_graphs = defaultdict(list)

    for graph in dataset:
        subject_to_graphs[graph.subject_id].append(graph)

    subject_ids = list(subject_to_graphs.keys())
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    split_idx = int(len(subject_ids) * train_ratio)

    if len(subject_ids) > 1:
        split_idx = max(1, min(split_idx, len(subject_ids) - 1))

    train_subjects = set(subject_ids[:split_idx])
    val_subjects = set(subject_ids[split_idx:])

    train_dataset = []
    val_dataset = []

    for subject_id, graphs in subject_to_graphs.items():
        if subject_id in train_subjects:
            train_dataset.extend(graphs)
        else:
            val_dataset.extend(graphs)

    return train_dataset, val_dataset


def move_batch_to_device(batch, device: torch.device):
    return batch.to(device)


def forward_model(model, batch) -> torch.Tensor:
    edge_attr = getattr(batch, "edge_attr", None)

    logits = model(
        x=batch.x,
        edge_index=batch.edge_index,
        batch=batch.batch,
        edge_attr=edge_attr,
    )
    return logits


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[int(t)][int(p)] += 1

    return cm


def add_confusion_matrices(cm_a: list[list[int]], cm_b: list[list[int]]) -> list[list[int]]:
    num_classes = len(cm_a)
    result = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for i in range(num_classes):
        for j in range(num_classes):
            result[i][j] = cm_a[i][j] + cm_b[i][j]

    return result


def metrics_from_confusion_matrix(confusion_matrix: list[list[int]]) -> dict:
    num_classes = len(confusion_matrix)
    total_samples = sum(sum(row) for row in confusion_matrix)
    correct = sum(confusion_matrix[i][i] for i in range(num_classes))
    accuracy = correct / max(1, total_samples)

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for cls in range(num_classes):
        tp = confusion_matrix[cls][cls]
        fp = sum(confusion_matrix[row][cls] for row in range(num_classes) if row != cls)
        fn = sum(confusion_matrix[cls][col] for col in range(num_classes) if col != cls)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class_precision[cls] = precision
        per_class_recall[cls] = recall
        per_class_f1[cls] = f1

    macro_precision = sum(per_class_precision.values()) / num_classes
    macro_recall = sum(per_class_recall.values()) / num_classes
    macro_f1 = sum(per_class_f1.values()) / num_classes

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": confusion_matrix,
        "num_samples": total_samples,
    }


def evaluate_model(model, data_loader, device: torch.device, num_classes: int) -> dict:
    model.eval()

    running_confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)

            logits = forward_model(model, batch)
            preds = logits.argmax(dim=1)
            targets = batch.y.view(-1)

            batch_cm = compute_confusion_matrix(preds, targets, num_classes=num_classes)
            running_confusion_matrix = add_confusion_matrices(running_confusion_matrix, batch_cm)

    return metrics_from_confusion_matrix(running_confusion_matrix)


def save_checkpoint(
    save_path: str,
    model,
    model_name: str,
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, save_path)


def count_model_parameters(model) -> dict:
    """Returns total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def get_class_distribution(dataset: list, num_classes: int) -> dict:
    """Returns per-class graph counts."""
    counts = {i: 0 for i in range(num_classes)}
    for graph in dataset:
        label = int(graph.y.item())
        counts[label] = counts.get(label, 0) + 1
    return counts


def train_gnn(
    epochs: int,
    learning_rate: float,
    batch_size: int,
    dataset_path: str,
    model_name: str = "GCN",
    hidden_channels: int = 64,
    dropout: float = 0.5,
    train_ratio: float = 0.8,
    save_mode: str = "best",
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str | None = None,
    num_node_features: int = 4,
    gat_heads_first_layer: int = 4,
    gat_heads_second_layer: int = 1,
    seed: int = 42,
    window_duration: float | None = None,
    plv_threshold: float | None = None,
    classification_mode: str = "binary",
    weight_decay: float = 1e-4,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    connectivity_type: str = "functional",
):
    """
    Enhanced training with:
    - Learning rate scheduler (ReduceLROnPlateau)
    - Weight decay (L2 regularization)
    - Train accuracy tracking
    - Comprehensive training report at end

    Supports:
    - binary      => C vs A
    - three_class => C vs A vs F
    """
    model_name = normalize_model_name(model_name)
    classification_mode = normalize_classification_mode(classification_mode)

    save_mode = str(save_mode).strip().lower()
    valid_save_modes = {"best", "last", "both"}
    if save_mode not in valid_save_modes:
        raise ValueError(f"save_mode must be one of {valid_save_modes}")

    class_config = get_class_configuration(classification_mode)
    num_classes = len(class_config["class_labels"])

    set_random_seed(seed)
    device = get_device()

    dataset = load_dataset(dataset_path)
    validate_dataset_metadata(dataset)

    train_dataset, val_dataset = split_dataset_by_subject(
        dataset=dataset,
        train_ratio=train_ratio,
        seed=seed,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            "Train/validation split resulted in an empty set. "
            "Check dataset size and subject distribution."
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(
        model_name=model_name,
        num_node_features=num_node_features,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        dropout=dropout,
        gat_heads_first_layer=gat_heads_first_layer,
        gat_heads_second_layer=gat_heads_second_layer,
    ).to(device)

    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler - reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=1e-6,
    )

    # Compute class weights (inverse frequency) to handle class imbalance
    train_labels = [int(graph.y.item()) for graph in train_dataset]
    num_classes_in_data = max(train_labels) + 1
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=num_classes_in_data).float()
    # Avoid division by zero for classes with no samples
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = (1.0 / class_counts) * len(train_labels) / num_classes_in_data
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    dataset_subjects = sorted({graph.subject_id for graph in dataset})
    train_subjects = sorted({graph.subject_id for graph in train_dataset})
    val_subjects = sorted({graph.subject_id for graph in val_dataset})

    # Class distribution analysis
    train_class_dist = get_class_distribution(train_dataset, num_classes)
    val_class_dist = get_class_distribution(val_dataset, num_classes)

    # Model info
    param_info = count_model_parameters(model)

    config = {
        "model_name": model_name,
        "classification_mode": classification_mode,
        "active_groups": list(class_config["groups"]),
        "class_labels": class_config["class_labels"],
        "num_classes": num_classes,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "batch_size": batch_size,
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "train_ratio": train_ratio,
        "save_mode": save_mode,
        "num_node_features": num_node_features,
        "gat_heads_first_layer": gat_heads_first_layer,
        "gat_heads_second_layer": gat_heads_second_layer,
        "seed": seed,
        "dataset_path": dataset_path,
        "window_duration": window_duration,
        "plv_threshold": plv_threshold,
        "connectivity_type": connectivity_type,
        "device": str(device),
        "num_total_graphs": len(dataset),
        "num_train_graphs": len(train_dataset),
        "num_val_graphs": len(val_dataset),
        "num_total_subjects": len(dataset_subjects),
        "num_train_subjects": len(train_subjects),
        "num_val_subjects": len(val_subjects),
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "train_class_distribution": train_class_dist,
        "val_class_distribution": val_class_dist,
        "model_parameters": param_info,
    }

    if checkpoint_name is None or not str(checkpoint_name).strip():
        checkpoint_name = f"{model_name.lower()}_{classification_mode}"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    best_metrics = None

    # For comprehensive report
    training_start_time = time.time()
    lr_changes = []
    all_train_losses = []
    all_val_losses = []
    all_val_accs = []
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_graphs = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()

            logits = forward_model(model, batch)
            targets = batch.y.view(-1)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # Track train accuracy
            preds = logits.argmax(dim=1)
            total_train_correct += (preds == targets).sum().item()

            num_graphs = batch.num_graphs
            total_train_loss += loss.item() * num_graphs
            total_train_graphs += num_graphs

        avg_train_loss = total_train_loss / max(1, total_train_graphs)
        train_accuracy = total_train_correct / max(1, total_train_graphs)

        # Validation
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
        )
        val_acc = val_metrics["accuracy"]

        # Compute validation loss
        model.eval()
        total_val_loss = 0.0
        total_val_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                logits = forward_model(model, batch)
                targets = batch.y.view(-1)
                vloss = criterion(logits, targets)
                total_val_loss += vloss.item() * batch.num_graphs
                total_val_graphs += batch.num_graphs
        avg_val_loss = total_val_loss / max(1, total_val_graphs)

        # Learning rate scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            lr_changes.append({"epoch": epoch, "old_lr": current_lr, "new_lr": new_lr})

        # Track history
        all_train_losses.append(avg_train_loss)
        all_val_losses.append(avg_val_loss)
        all_val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(val_metrics)
            epochs_without_improvement = 0

            if save_mode in {"best", "both"}:
                best_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_best.pt")
                save_checkpoint(
                    save_path=best_path,
                    model=model,
                    model_name=model_name,
                    epoch=epoch,
                    metrics=val_metrics,
                    config=config,
                )
        else:
            epochs_without_improvement += 1

        epoch_duration = time.time() - epoch_start

        epoch_info = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_confusion_matrix": val_metrics["confusion_matrix"],
            "best_val_accuracy_so_far": best_val_acc,
            "best_epoch_so_far": best_epoch,
            "learning_rate": new_lr,
            "epoch_time_sec": epoch_duration,
        }

        yield epoch_info

    # Save last checkpoint
    if save_mode in {"last", "both"}:
        last_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_last.pt")
        final_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
        )
        save_checkpoint(
            save_path=last_path,
            model=model,
            model_name=model_name,
            epoch=epochs,
            metrics=final_metrics,
            config=config,
        )

    if save_mode in {"best", "both"} and best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_training_time = time.time() - training_start_time

    # Compute overfitting indicators
    final_train_loss = all_train_losses[-1] if all_train_losses else 0
    final_val_loss = all_val_losses[-1] if all_val_losses else 0
    overfit_gap = final_val_loss - final_train_loss

    # Compute convergence info
    loss_improvement_first_half = (
        (all_train_losses[0] - all_train_losses[len(all_train_losses) // 2])
        if len(all_train_losses) > 1 else 0
    )
    loss_improvement_second_half = (
        (all_train_losses[len(all_train_losses) // 2] - all_train_losses[-1])
        if len(all_train_losses) > 1 else 0
    )

    # Comprehensive final summary
    final_summary = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "config": config,
        "training_report": {
            "total_time_seconds": round(total_training_time, 1),
            "avg_epoch_time_seconds": round(total_training_time / max(1, epochs), 2),
            "total_epochs_run": epochs,
            "best_epoch": best_epoch,
            "epochs_without_improvement_at_end": epochs_without_improvement,
            "model_parameters": param_info,
            "learning_rate_changes": lr_changes,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
            "initial_learning_rate": learning_rate,
            "final_train_loss": round(final_train_loss, 5),
            "final_val_loss": round(final_val_loss, 5),
            "best_val_loss": round(best_val_loss, 5),
            "overfitting_gap": round(overfit_gap, 5),
            "loss_improvement_first_half": round(loss_improvement_first_half, 5),
            "loss_improvement_second_half": round(loss_improvement_second_half, 5),
            "train_class_distribution": train_class_dist,
            "val_class_distribution": val_class_dist,
            "convergence_analysis": _analyze_convergence(
                all_train_losses, all_val_losses, all_val_accs, best_epoch, epochs
            ),
        },
    }

    yield final_summary


def _analyze_convergence(
    train_losses: list,
    val_losses: list,
    val_accs: list,
    best_epoch: int,
    total_epochs: int,
) -> dict:
    """Generates a human-readable convergence analysis."""
    analysis = {}

    if not train_losses:
        return {"message": "No training data to analyze."}

    # Check if model converged
    last_10_pct = max(1, len(train_losses) // 10)
    recent_loss_std = _std(train_losses[-last_10_pct:])
    recent_val_std = _std(val_losses[-last_10_pct:])

    analysis["loss_std_last_10pct"] = round(recent_loss_std, 5)
    analysis["val_loss_std_last_10pct"] = round(recent_val_std, 5)

    # Detect plateau
    if recent_loss_std < 0.01:
        analysis["plateau_detected"] = True
        analysis["plateau_note"] = (
            "Training loss has plateaued (std < 0.01 in last 10% of epochs). "
            "The model may need: lower learning rate, more capacity, better features, "
            "or the data may be inherently noisy/difficult."
        )
    else:
        analysis["plateau_detected"] = False

    # Detect overfitting
    if len(val_losses) > 10:
        avg_val_early = sum(val_losses[:10]) / 10
        avg_val_late = sum(val_losses[-10:]) / 10
        avg_train_late = sum(train_losses[-10:]) / 10

        if avg_val_late > avg_val_early and avg_train_late < avg_val_late:
            analysis["overfitting_detected"] = True
            analysis["overfitting_note"] = (
                "Validation loss is increasing while training loss is decreasing. "
                "This indicates overfitting. Consider: more dropout, less epochs, "
                "more data augmentation, or weight decay increase."
            )
        else:
            analysis["overfitting_detected"] = False

    # Best epoch position
    best_position = best_epoch / total_epochs
    if best_position < 0.3:
        analysis["early_peak_note"] = (
            f"Best result was at epoch {best_epoch}/{total_epochs} (early 30%). "
            "The model may be overfitting after that. Try fewer epochs or stronger regularization."
        )
    elif best_position > 0.9:
        analysis["late_peak_note"] = (
            f"Best result at epoch {best_epoch}/{total_epochs} (last 10%). "
            "The model may still be improving. Consider training for more epochs."
        )

    # Overall assessment
    best_acc = max(val_accs) if val_accs else 0
    if best_acc < 0.55:
        analysis["performance_assessment"] = (
            "Performance is near random chance. Possible causes: "
            "1) Insufficient node features (only 4 band powers). "
            "2) PLV threshold too high causing very sparse/disconnected graphs. "
            "3) Time window too short for meaningful spectral features. "
            "4) Class imbalance in training data. "
            "5) The signal differences between classes are subtle and require more data."
        )
    elif best_acc < 0.70:
        analysis["performance_assessment"] = (
            "Moderate performance. The model learns some patterns but struggles. "
            "Try: lower PLV threshold (0.7-0.8), longer time windows (8-10s), "
            "or include more frequency bands as node features."
        )
    elif best_acc < 0.85:
        analysis["performance_assessment"] = (
            "Good performance. The model captures meaningful EEG connectivity patterns. "
            "Fine-tuning hyperparameters may yield further improvements."
        )
    else:
        analysis["performance_assessment"] = (
            "Excellent performance. Verify no data leakage between train/val subjects."
        )

    return analysis


def _std(values: list) -> float:
    """Standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    dataset_path = os.path.join(project_root, "data", "eeg_graphs_dataset.pt")
    checkpoint_dir = os.path.join(project_root, "checkpoints")

    for info in train_gnn(
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
        dataset_path=dataset_path,
        model_name="GCN",
        hidden_channels=64,
        dropout=0.5,
        train_ratio=0.8,
        save_mode="both",
        checkpoint_dir=checkpoint_dir,
        checkpoint_name="debug_gcn_3class",
        num_node_features=4,
        gat_heads_first_layer=4,
        gat_heads_second_layer=1,
        seed=42,
        window_duration=5.0,
        plv_threshold=0.9,
        classification_mode="three_class",
    ):
        print(info)

