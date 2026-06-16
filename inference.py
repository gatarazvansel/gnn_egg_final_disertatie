from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.preprocessing import (
    load_and_preprocess_eeg,
    extract_time_windows,
    time_windows_to_numpy,
)
from src.graph_builder import build_graphs_from_time_windows
from src.models.model_factory import get_model, normalize_model_name


DEFAULT_CLASS_LABELS = {
    "binary": {
        0: "Healthy Control",
        1: "Alzheimer's Disease",
    },
    "three_class": {
        0: "Healthy Control",
        1: "Alzheimer's Disease",
        2: "Frontotemporal Dementia",
    },
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: str, map_location: str | None = None) -> dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if map_location is None:
        map_location = "cpu"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    required_keys = {"model_state_dict", "model_name", "config"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"Invalid checkpoint. Missing keys: {sorted(missing_keys)}"
        )

    return checkpoint


def get_class_labels_from_checkpoint(checkpoint: dict) -> dict[int, str]:
    config = checkpoint["config"]

    if "class_labels" in config:
        raw_labels = config["class_labels"]
        return {int(k): str(v) for k, v in raw_labels.items()}

    classification_mode = config.get("classification_mode", "binary")
    fallback = DEFAULT_CLASS_LABELS.get(classification_mode)
    if fallback is None:
        raise ValueError(
            "Could not determine class labels from checkpoint config."
        )
    return fallback


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    config = checkpoint["config"]
    model_name = normalize_model_name(checkpoint["model_name"])

    model = get_model(
        model_name=model_name,
        num_node_features=config.get("num_node_features", 4),
        hidden_channels=config.get("hidden_channels", 64),
        num_classes=config.get("num_classes", 2),
        dropout=config.get("dropout", 0.5),
        gat_heads_first_layer=config.get("gat_heads_first_layer", 4),
        gat_heads_second_layer=config.get("gat_heads_second_layer", 1),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def validate_inference_settings(window_duration: float, threshold: float) -> None:
    if window_duration <= 0:
        raise ValueError("window_duration must be > 0.")

    if threshold < 0:
        raise ValueError("threshold must be >= 0.")


def predict_graphs(
    model,
    graphs: list,
    batch_size: int,
    device: torch.device,
    class_labels: dict[int, str],
) -> list[dict[str, Any]]:
    if len(graphs) == 0:
        raise ValueError("No graphs provided for inference.")

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    window_results = []
    global_window_index = 0
    sorted_class_ids = sorted(class_labels.keys())

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
                edge_attr=getattr(batch, "edge_attr", None),
            )

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(batch.num_graphs):
                pred_class = int(preds[i].item())
                confidence = float(probs[i, pred_class].item())

                class_probabilities = {
                    class_labels[class_id]: float(probs[i, class_id].item())
                    for class_id in sorted_class_ids
                }

                window_results.append(
                    {
                        "window_index": global_window_index,
                        "predicted_class": pred_class,
                        "predicted_label": class_labels[pred_class],
                        "confidence": confidence,
                        "class_probabilities": class_probabilities,
                    }
                )
                global_window_index += 1

    return window_results


def aggregate_window_predictions(
    window_results: list[dict[str, Any]],
    class_labels: dict[int, str],
) -> dict[str, Any]:
    if len(window_results) == 0:
        raise ValueError("window_results is empty.")

    num_windows = len(window_results)
    class_names = [class_labels[idx] for idx in sorted(class_labels.keys())]

    mean_probabilities = {}
    window_counts = {}
    window_ratios = {}

    for class_name in class_names:
        probs = [row["class_probabilities"][class_name] for row in window_results]
        mean_probabilities[class_name] = sum(probs) / num_windows

        count = sum(1 for row in window_results if row["predicted_label"] == class_name)
        window_counts[class_name] = count
        window_ratios[class_name] = count / num_windows

    final_label = max(mean_probabilities, key=mean_probabilities.get)
    final_class = next(idx for idx, name in class_labels.items() if name == final_label)
    final_confidence = mean_probabilities[final_label]

    return {
        "final_class": final_class,
        "final_label": final_label,
        "final_confidence": final_confidence,
        "num_windows": num_windows,
        "mean_probabilities": mean_probabilities,
        "window_counts": window_counts,
        "window_ratios": window_ratios,
    }


def prepare_window_time_metadata(
    window_results: list[dict[str, Any]],
    window_duration: float,
) -> list[dict[str, Any]]:
    enriched = []

    for result in window_results:
        window_index = result["window_index"]
        start_sec = window_index * window_duration
        end_sec = start_sec + window_duration

        enriched_result = dict(result)
        enriched_result["start_sec"] = start_sec
        enriched_result["end_sec"] = end_sec

        enriched.append(enriched_result)

    return enriched


def run_recording_inference(
    eeg_file_path: str,
    checkpoint_path: str,
    window_duration: float,
    threshold: float,
    batch_size: int = 32,
    apply_filter: bool = True,
    apply_ica: bool = False,
    connectivity_type: str | None = None,
) -> dict[str, Any]:
    validate_inference_settings(window_duration=window_duration, threshold=threshold)

    device = get_device()

    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    class_labels = get_class_labels_from_checkpoint(checkpoint)
    model = build_model_from_checkpoint(checkpoint, device=device)

    # Resolve the connectivity type: explicit argument overrides the value stored
    # in the checkpoint config; otherwise fall back to what the model was trained on.
    if connectivity_type is None:
        connectivity_type = checkpoint["config"].get("connectivity_type", "functional")

    raw_signal = load_and_preprocess_eeg(
        file_path=eeg_file_path,
        apply_filter=apply_filter,
        apply_ica=apply_ica,
    )

    mne_epochs = extract_time_windows(
        raw=raw_signal,
        window_duration=window_duration,
        overlap=0.0,
        reject_amplitude_uv=None,
    )
    windows_array = time_windows_to_numpy(mne_epochs)

    if len(windows_array) == 0:
        raise ValueError("No time windows could be extracted from the EEG recording.")

    graphs = build_graphs_from_time_windows(
        windows_array=windows_array,
        sfreq=raw_signal.info["sfreq"],
        threshold=threshold,
        label=None,
        subject_id="diagnosis_recording",
        channel_names=raw_signal.ch_names,
        connectivity_type=connectivity_type,
    )

    window_results = predict_graphs(
        model=model,
        graphs=graphs,
        batch_size=batch_size,
        device=device,
        class_labels=class_labels,
    )
    window_results = prepare_window_time_metadata(
        window_results=window_results,
        window_duration=window_duration,
    )

    summary = aggregate_window_predictions(
        window_results=window_results,
        class_labels=class_labels,
    )

    inference_result = {
        "summary": summary,
        "window_results": window_results,
        "raw_signal": raw_signal,
        "mne_epochs_object": mne_epochs,
        "windows_array": windows_array,
        "channel_names": raw_signal.ch_names,
        "sfreq": raw_signal.info["sfreq"],
        "checkpoint_info": {
            "model_name": checkpoint["model_name"],
            "config": checkpoint["config"],
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics"),
            "class_labels": class_labels,
            "connectivity_type": connectivity_type,
        },
    }

    return inference_result


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    sample_eeg = os.path.join(
        project_root,
        "data",
        "holdout_diagnosis",
        "sub-024",
        "eeg",
        "sub-024_task-eyesclosed_eeg.set",
    )

    sample_checkpoint = os.path.join(
        project_root,
        "checkpoints",
        "gcn_three_class_best.pt",
    )

    if os.path.exists(sample_eeg) and os.path.exists(sample_checkpoint):
        result = run_recording_inference(
            eeg_file_path=sample_eeg,
            checkpoint_path=sample_checkpoint,
            window_duration=5.0,
            threshold=0.9,
            batch_size=16,
            apply_filter=True,
            apply_ica=False,
        )

        print("\n=== INFERENCE SUMMARY ===")
        print(result["summary"])

        print("\n=== FIRST 3 WINDOW RESULTS ===")
        for row in result["window_results"][:3]:
            print(row)
    else:
        print("Sample EEG file or checkpoint not found for inference test.")