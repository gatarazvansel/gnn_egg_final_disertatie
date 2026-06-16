from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch

from src.preprocessing import (
    load_and_preprocess_eeg,
    extract_time_windows,
    time_windows_to_numpy,
)
from src.graph_builder import build_graphs_from_time_windows


CLASS_CONFIGS = {
    "binary": {
        "groups": ("C", "A"),
        "label_mapping": {"C": 0, "A": 1},
        "class_labels": {
            0: "Healthy Control",
            1: "Alzheimer's Disease",
        },
    },
    "three_class": {
        "groups": ("C", "A", "F"),
        "label_mapping": {"C": 0, "A": 1, "F": 2},
        "class_labels": {
            0: "Healthy Control",
            1: "Alzheimer's Disease",
            2: "Frontotemporal Dementia",
        },
    },
}


def normalize_classification_mode(classification_mode: str) -> str:
    if classification_mode is None:
        raise ValueError("classification_mode cannot be None.")

    mode = str(classification_mode).strip().lower()
    if mode not in CLASS_CONFIGS:
        raise ValueError(
            f"Unsupported classification_mode '{classification_mode}'. "
            f"Supported modes: {tuple(CLASS_CONFIGS.keys())}"
        )
    return mode


def get_class_configuration(classification_mode: str) -> dict:
    mode = normalize_classification_mode(classification_mode)
    return CLASS_CONFIGS[mode]


def validate_participants_table(df: pd.DataFrame) -> None:
    required_columns = {"participant_id", "Group"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"participants.tsv is missing required columns: {sorted(missing_columns)}"
        )


def build_eeg_file_path(base_data_dir: str, subject_id: str) -> str:
    return os.path.join(
        base_data_dir,
        "derivatives",
        subject_id,
        "eeg",
        f"{subject_id}_task-eyesclosed_eeg.set",
    )


def parse_label(group_value: str, classification_mode: str) -> int | None:
    config = get_class_configuration(classification_mode)

    if pd.isna(group_value):
        return None

    group_value = str(group_value).strip().upper()
    return config["label_mapping"].get(group_value)


def build_graphs_for_subject(
    file_path: str,
    subject_id: str,
    label: int,
    window_duration: float,
    threshold: float,
    apply_filter: bool = True,
    apply_ica: bool = False,
    overlap: float = 0.0,
    connectivity_type: str = "functional",
) -> list:
    raw_signal = load_and_preprocess_eeg(
        file_path=file_path,
        apply_filter=apply_filter,
        apply_ica=apply_ica,
    )

    mne_epochs = extract_time_windows(
        raw=raw_signal,
        window_duration=window_duration,
        overlap=overlap,
        reject_amplitude_uv=None,
    )

    windows_array = time_windows_to_numpy(mne_epochs)

    graphs = build_graphs_from_time_windows(
        windows_array=windows_array,
        sfreq=raw_signal.info["sfreq"],
        threshold=threshold,
        label=label,
        subject_id=subject_id,
        channel_names=raw_signal.ch_names,
        connectivity_type=connectivity_type,
    )

    return graphs


def augment_graphs_with_noise(
    graphs: list,
    noise_std: float = 0.01,
    num_augmented_copies: int = 1,
) -> list:
    """
    Creates augmented copies of graphs by adding Gaussian noise to node features.

    Parameters
    ----------
    graphs : list
        Original list of PyG Data objects.
    noise_std : float
        Standard deviation of Gaussian noise (relative to feature std).
    num_augmented_copies : int
        Number of noisy copies to generate per original graph.

    Returns
    -------
    list
        Augmented graphs (does NOT include originals).
    """
    augmented = []

    for graph in graphs:
        for _ in range(num_augmented_copies):
            noisy_graph = graph.clone()
            # Scale noise relative to each feature's standard deviation
            feature_std = noisy_graph.x.std(dim=0, keepdim=True).clamp(min=1e-8)
            noise = torch.randn_like(noisy_graph.x) * feature_std * noise_std
            noisy_graph.x = noisy_graph.x + noise
            augmented.append(noisy_graph)

    return augmented


def build_pyg_dataset(
    tsv_path: str,
    base_data_dir: str,
    output_path: str,
    threshold: float = 0.9,
    window_duration: float = 5.0,
    apply_filter: bool = True,
    apply_ica: bool = False,
    classification_mode: str = "binary",
    overlap: float = 0.0,
    noise_augmentation: bool = False,
    noise_std: float = 0.01,
    noise_copies: int = 1,
    connectivity_type: str = "functional",
) -> list:
    """
    Builds a list of PyTorch Geometric Data objects from participants.tsv.

    classification_mode:
    - "binary"      => C vs A
    - "three_class" => C vs A vs F

    connectivity_type:
    - "functional"  => edges from PLV synchronization (data-driven)
    - "topological" => edges from the physical 10-20 electrode layout (fixed)
    """
    classification_mode = normalize_classification_mode(classification_mode)
    class_config = get_class_configuration(classification_mode)

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"participants.tsv not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    validate_participants_table(df)

    dataset_list = []
    included_subjects = []
    skipped_subjects = []
    failed_subjects = []
    graph_counter_by_group = Counter()
    subject_counter_by_group = Counter()

    print("=" * 70)
    print("STARTING EEG GRAPH DATASET GENERATION")
    print(f"TSV file            : {tsv_path}")
    print(f"Base data dir       : {base_data_dir}")
    print(f"Output path         : {output_path}")
    print(f"Window duration     : {window_duration}s")
    print(f"Window overlap      : {overlap * 100:.0f}%")
    print(f"PLV threshold       : {threshold}")
    print(f"Apply filter        : {apply_filter}")
    print(f"Apply ICA           : {apply_ica}")
    print(f"Classification mode : {classification_mode}")
    print(f"Connectivity type   : {connectivity_type}")
    print(f"Groups kept         : {class_config['groups']}")
    print(f"Noise augmentation  : {noise_augmentation} (std={noise_std}, copies={noise_copies})")
    print("=" * 70)

    for _, row in df.iterrows():
        subject_id = str(row["participant_id"]).strip()
        group = str(row["Group"]).strip().upper()

        if group not in class_config["groups"]:
            skipped_subjects.append((subject_id, f"group '{group}' excluded"))
            print(f"[SKIP] {subject_id}: group '{group}' excluded for mode '{classification_mode}'")
            continue

        label = parse_label(group, classification_mode=classification_mode)
        if label is None:
            skipped_subjects.append((subject_id, f"unsupported group '{group}'"))
            print(f"[SKIP] {subject_id}: unsupported group '{group}'")
            continue

        file_path = build_eeg_file_path(base_data_dir, subject_id)

        if not os.path.exists(file_path):
            skipped_subjects.append((subject_id, "missing EEG file"))
            print(f"[SKIP] {subject_id}: file not found -> {file_path}")
            continue

        print(f"\n[PROCESSING] {subject_id} | Group={group} | Label={label}")

        try:
            subject_graphs = build_graphs_for_subject(
                file_path=file_path,
                subject_id=subject_id,
                label=label,
                window_duration=window_duration,
                threshold=threshold,
                apply_filter=apply_filter,
                apply_ica=apply_ica,
                overlap=overlap,
                connectivity_type=connectivity_type,
            )

            if len(subject_graphs) == 0:
                skipped_subjects.append((subject_id, "no time windows generated"))
                print(f"[SKIP] {subject_id}: no valid time windows generated")
                continue

            dataset_list.extend(subject_graphs)
            included_subjects.append(subject_id)
            graph_counter_by_group[group] += len(subject_graphs)
            subject_counter_by_group[group] += 1

            print(f"[OK] {subject_id}: generated {len(subject_graphs)} graphs")

        except Exception as exc:
            failed_subjects.append((subject_id, str(exc)))
            print(f"[ERROR] {subject_id}: {exc}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Apply Gaussian noise augmentation if enabled
    num_original_graphs = len(dataset_list)
    if noise_augmentation and num_original_graphs > 0:
        print(f"\n[AUGMENTATION] Generating {noise_copies} noisy copy/copies per graph (std={noise_std})...")
        augmented_graphs = augment_graphs_with_noise(
            graphs=dataset_list,
            noise_std=noise_std,
            num_augmented_copies=noise_copies,
        )
        dataset_list.extend(augmented_graphs)
        print(f"[AUGMENTATION] Added {len(augmented_graphs)} augmented graphs")

    torch.save(dataset_list, output_path)

    print("\n" + "=" * 70)
    print("DATASET GENERATION FINISHED")
    print(f"Included subjects : {len(included_subjects)}")
    print(f"Skipped subjects  : {len(skipped_subjects)}")
    print(f"Failed subjects   : {len(failed_subjects)}")
    print(f"Original graphs   : {num_original_graphs}")
    if noise_augmentation:
        print(f"Augmented graphs  : {len(dataset_list) - num_original_graphs}")
    print(f"Total graphs      : {len(dataset_list)}")
    for group in class_config["groups"]:
        print(f"Subjects in {group} : {subject_counter_by_group.get(group, 0)}")
        print(f"Graphs in {group}   : {graph_counter_by_group.get(group, 0)}")
    print(f"Saved to          : {output_path}")
    print("=" * 70)

    if skipped_subjects:
        print("\nSkipped subjects summary:")
        for subject_id, reason in skipped_subjects:
            print(f" - {subject_id}: {reason}")

    if failed_subjects:
        print("\nFailed subjects summary:")
        for subject_id, reason in failed_subjects:
            print(f" - {subject_id}: {reason}")

    return dataset_list


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    tsv_file = os.path.join(project_root, "data", "participants_train.tsv")
    base_dir = os.path.join(project_root, "data")
    output_file = os.path.join(project_root, "data", "eeg_graphs_dataset.pt")

    build_pyg_dataset(
        tsv_path=tsv_file,
        base_data_dir=base_dir,
        output_path=output_file,
        threshold=0.9,
        window_duration=5.0,
        apply_filter=True,
        apply_ica=False,
        classification_mode="three_class",
    )