from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.models.model_factory import get_supported_models


CLASSIFICATION_MODE_LABELS = {
    "binary": "Binary (C vs A)",
    "three_class": "Three-class (C vs A vs F)",
}

CONNECTIVITY_TYPE_LABELS = {
    "functional": "Functional (PLV synchronization)",
    "topological": "Topological (10-20 electrode layout)",
}


def setup_page():
    st.set_page_config(
        page_title="EEG GNN Alzheimer Detection",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Pastel purple theme
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background-color: #f8f5fc;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #ede7f6;
            border-radius: 10px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            color: #4a148c;
        }
        .stTabs [aria-selected="true"] {
            background-color: #b39ddb !important;
            color: white !important;
            border-radius: 8px;
        }

        /* Headers */
        h1, h2, h3 {
            color: #4a148c !important;
        }
        h4, h5 {
            color: #6a1b9a !important;
        }

        /* Buttons */
        .stButton > button[kind="primary"] {
            background-color: #7e57c2 !important;
            border: none;
            color: white !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #5e35b1 !important;
        }
        .stButton > button {
            border-color: #b39ddb !important;
            color: #4a148c !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #4a148c;
        }

        /* Slider */
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: #7e57c2;
        }

        /* Info/success/error boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Expander */
        .streamlit-expanderHeader {
            color: #4a148c !important;
            font-weight: 600;
        }

        /* Dividers */
        hr {
            border-color: #d1c4e9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_available_subjects(base_data_dir: str) -> list[str]:
    derivatives_dir = os.path.join(base_data_dir, "derivatives")

    if not os.path.exists(derivatives_dir):
        return []

    subjects = []

    for entry in sorted(os.listdir(derivatives_dir)):
        subject_dir = os.path.join(derivatives_dir, entry)
        eeg_file = os.path.join(
            subject_dir,
            "eeg",
            f"{entry}_task-eyesclosed_eeg.set",
        )

        if os.path.isdir(subject_dir) and os.path.exists(eeg_file):
            subjects.append(entry)

    return subjects


def get_available_holdout_subjects(holdout_dir: str) -> list[str]:
    if not os.path.exists(holdout_dir):
        return []

    subjects = []

    for entry in sorted(os.listdir(holdout_dir)):
        subject_dir = os.path.join(holdout_dir, entry)
        eeg_file = os.path.join(
            subject_dir,
            "eeg",
            f"{entry}_task-eyesclosed_eeg.set",
        )

        if os.path.isdir(subject_dir) and os.path.exists(eeg_file):
            subjects.append(entry)

    return subjects


def get_available_checkpoints(checkpoint_dir: str) -> list[str]:
    if not os.path.exists(checkpoint_dir):
        return []

    return [
        file_name
        for file_name in sorted(os.listdir(checkpoint_dir))
        if file_name.endswith(".pt")
    ]


# ─────────────────────────────────────────────────────────────
# TAB 1: EEG Signal Exploration Controls
# ─────────────────────────────────────────────────────────────

def render_signal_controls(base_data_dir: str) -> dict[str, Any]:
    """Renders inline controls for the EEG Signal Exploration tab."""
    st.markdown("#### Signal Exploration Settings")

    available_subjects = get_available_subjects(base_data_dir)

    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

    with col1:
        if available_subjects:
            selected_subject = st.selectbox(
                "Subject",
                available_subjects,
                index=0,
                help=(
                    "Select a subject from the dataset to visualize their preprocessed EEG recording. "
                    "Each subject corresponds to one EEGLAB .set file in data/derivatives/."
                ),
                key="signal_subject",
            )
        else:
            selected_subject = None
            st.warning("No subjects found in data/derivatives/.")

    with col2:
        window_duration = st.slider(
            "Display Duration (s)",
            min_value=2.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help=(
                "Duration of the EEG segment to display (in seconds). "
                "This controls how many seconds of the recording are shown in the time-series plot."
            ),
            key="signal_window_duration",
        )

    with col3:
        apply_filter = st.checkbox(
            "Band-pass Filter",
            value=True,
            help=(
                "Apply a FIR band-pass filter between 0.5 Hz and 45 Hz. "
                "This removes slow baseline drifts (< 0.5 Hz) and high-frequency noise/power-line artifacts (> 45 Hz)."
            ),
            key="signal_filter",
        )

    with col4:
        apply_ica = st.checkbox(
            "ICA Cleaning",
            value=False,
            help=(
                "Apply Independent Component Analysis (ICA) to remove eye-blink and muscle artifacts. "
                "ICA decomposes the signal into independent sources and removes those associated with artifacts. "
                "Note: enabling ICA significantly increases preprocessing time."
            ),
            key="signal_ica",
        )

    st.markdown("---")

    return {
        "selected_subject": selected_subject,
        "window_duration": window_duration,
        "apply_filter": apply_filter,
        "apply_ica": apply_ica,
    }


# ─────────────────────────────────────────────────────────────
# TAB 2: Brain Graph Controls
# ─────────────────────────────────────────────────────────────

def render_graph_controls(base_data_dir: str) -> dict[str, Any]:
    """Renders inline controls for the Brain Graph tab."""
    st.markdown("#### Connectivity Graph Settings")

    available_subjects = get_available_subjects(base_data_dir)

    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])

    with col1:
        if available_subjects:
            selected_subject = st.selectbox(
                "Subject",
                available_subjects,
                index=0,
                help=(
                    "Select a subject whose EEG recording will be used to compute functional connectivity. "
                    "The PLV matrix is computed from the first N seconds of the preprocessed recording."
                ),
                key="graph_subject",
            )
        else:
            selected_subject = None
            st.warning("No subjects found in data/derivatives/.")

    with col2:
        window_duration = st.slider(
            "Analysis Window (s)",
            min_value=2.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help=(
                "Duration of the signal segment (in seconds) used for PLV computation. "
                "Longer windows give more stable PLV estimates but represent a broader time range."
            ),
            key="graph_window_duration",
        )

    with col3:
        connectivity_type = st.selectbox(
            "Connectivity Type",
            options=list(CONNECTIVITY_TYPE_LABELS.keys()),
            format_func=lambda x: CONNECTIVITY_TYPE_LABELS[x],
            index=0,
            help=(
                "How graph edges between electrodes are created:\n"
                "- Functional (PLV): data-driven edges based on phase synchronization "
                "between channels in the alpha band. Edges change with the EEG signal.\n"
                "- Topological (10-20): fixed edges based on the physical distance between "
                "electrodes on the scalp. The same graph structure is used for every window "
                "(only node features change)."
            ),
            key="graph_connectivity_type",
        )

    with col4:
        apply_filter = st.checkbox(
            "Band-pass Filter",
            value=True,
            help="Apply FIR band-pass filter (0.5-45 Hz) before PLV computation.",
            key="graph_filter",
        )

    with col5:
        apply_ica = st.checkbox(
            "ICA Cleaning",
            value=False,
            help="Apply ICA artifact cleaning before PLV computation.",
            key="graph_ica",
        )

    if connectivity_type == "topological":
        threshold_label = "Proximity Threshold"
        threshold_help = (
            "Spatial proximity threshold for graph edge creation. "
            "Proximity is computed from the physical distance between electrodes "
            "(proximity = 1 - distance / max_distance), so values near 1 mean the "
            "electrodes are very close. Only electrode pairs with proximity >= threshold "
            "are connected. Higher values keep only the closest (neighboring) electrodes."
        )
    else:
        threshold_label = "PLV Threshold"
        threshold_help = (
            "Phase Locking Value (PLV) threshold for graph edge creation. "
            "PLV measures the synchronization between two channels: PLV = |mean(e^(j*delta_phi))|, where delta_phi is the "
            "instantaneous phase difference. Only channel pairs with PLV >= threshold will be connected "
            "by an edge in the brain graph. Higher values produce sparser graphs with only the strongest connections."
        )

    plv_threshold = st.slider(
        threshold_label,
        min_value=0.1,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help=threshold_help,
        key="graph_plv_threshold",
    )

    st.markdown("---")

    return {
        "selected_subject": selected_subject,
        "window_duration": window_duration,
        "plv_threshold": plv_threshold,
        "apply_filter": apply_filter,
        "apply_ica": apply_ica,
        "connectivity_type": connectivity_type,
    }


# ─────────────────────────────────────────────────────────────
# TAB 3: Model Training Controls
# ─────────────────────────────────────────────────────────────

def get_dataset_info(dataset_path: str) -> dict[str, Any] | None:
    """Load the dataset and extract summary information without keeping it in memory."""
    import torch

    if not os.path.exists(dataset_path):
        return None

    try:
        dataset = torch.load(dataset_path, weights_only=False)
        if not isinstance(dataset, list) or len(dataset) == 0:
            return None

        labels = [int(graph.y.item()) for graph in dataset]
        subjects = sorted(set(getattr(graph, "subject_id", "unknown") for graph in dataset))
        num_classes = max(labels) + 1

        from collections import Counter
        label_counts = Counter(labels)

        # Determine classification mode from num_classes
        if num_classes == 2:
            mode = "binary"
            class_names = {0: "Healthy Control", 1: "Alzheimer's Disease"}
        else:
            mode = "three_class"
            class_names = {0: "Healthy Control", 1: "Alzheimer's Disease", 2: "Frontotemporal Dementia"}

        class_distribution = {
            class_names.get(k, f"Class {k}"): v
            for k, v in sorted(label_counts.items())
        }

        # Try to get file modification time
        import time
        mod_time = os.path.getmtime(dataset_path)
        mod_time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))

        return {
            "num_graphs": len(dataset),
            "num_subjects": len(subjects),
            "num_classes": num_classes,
            "classification_mode": mode,
            "class_distribution": class_distribution,
            "modified": mod_time_str,
            "file_size_mb": round(os.path.getsize(dataset_path) / (1024 * 1024), 1),
        }
    except Exception:
        return None


def render_dataset_controls(dataset_path: str) -> dict[str, Any]:
    """Renders dataset generation controls with status awareness."""
    st.markdown("### Step 1: Dataset")

    # --- Show current dataset status ---
    dataset_info = get_dataset_info(dataset_path)

    if dataset_info is not None:
        st.success(
            f"Dataset loaded: **{dataset_info['num_graphs']} graphs** from "
            f"**{dataset_info['num_subjects']} subjects** | "
            f"**{dataset_info['num_classes']} classes** "
            f"({dataset_info['classification_mode']}) | "
            f"Last built: {dataset_info['modified']} | "
            f"Size: {dataset_info['file_size_mb']} MB"
        )

        # Show class distribution
        dist_cols = st.columns(dataset_info["num_classes"])
        for i, (class_name, count) in enumerate(dataset_info["class_distribution"].items()):
            with dist_cols[i]:
                pct = count / dataset_info["num_graphs"] * 100
                st.metric(class_name, f"{count} graphs", f"{pct:.1f}%")

        detected_mode = dataset_info["classification_mode"]
    else:
        st.warning(
            "No dataset found. You need to build a dataset before training. "
            "Configure the parameters below and click 'Build Dataset'."
        )
        detected_mode = None

    # --- Build new dataset (expander if dataset exists, always visible if not) ---
    build_expanded = dataset_info is None
    with st.expander(
        "Build New Dataset" if dataset_info else "Build Dataset",
        expanded=build_expanded,
    ):
        st.caption(
            "Generate a graph dataset from EEG recordings. "
            "Each recording is segmented into time windows, and each window becomes a brain graph. "
            "Building a new dataset will overwrite the existing one."
        )

        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])

        with col1:
            classification_mode = st.selectbox(
                "Classification Mode",
                options=list(CLASSIFICATION_MODE_LABELS.keys()),
                format_func=lambda x: CLASSIFICATION_MODE_LABELS[x],
                index=0,
                help=(
                    "Choose the classification task:\n"
                    "- Binary (C vs A): Healthy Controls vs Alzheimer's Disease\n"
                    "- Three-class (C vs A vs F): Healthy Controls vs Alzheimer's vs Frontotemporal Dementia\n\n"
                    "This determines which subjects from participants.tsv are included in the dataset."
                ),
                key="dataset_classification_mode",
            )

        with col2:
            window_duration = st.slider(
                "Time Window Duration (s)",
                min_value=2.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help=(
                    "Duration of each time window (in seconds) for signal segmentation. "
                    "The continuous EEG recording is split into segments of this length. "
                    "Each segment becomes one brain graph."
                ),
                key="dataset_window_duration",
            )

        with col3:
            plv_threshold = st.slider(
                "PLV Threshold",
                min_value=0.1,
                max_value=0.99,
                value=0.5,
                step=0.05,
                help=(
                    "Phase Locking Value threshold for the adjacency matrix. "
                    "Only electrode pairs with PLV >= threshold become edges. "
                    "Higher threshold produces sparser graphs."
                ),
                key="dataset_plv_threshold",
            )

        with col4:
            apply_filter = st.checkbox(
                "Band-pass Filter",
                value=True,
                help="Apply FIR band-pass filter (0.5-45 Hz) during preprocessing.",
                key="dataset_filter",
            )

        with col5:
            apply_ica = st.checkbox(
                "ICA Cleaning",
                value=False,
                help="Apply ICA artifact removal during preprocessing (slower).",
                key="dataset_ica",
            )

        # --- Connectivity strategy ---
        connectivity_type = st.selectbox(
            "Connectivity Type (graph edges)",
            options=list(CONNECTIVITY_TYPE_LABELS.keys()),
            format_func=lambda x: CONNECTIVITY_TYPE_LABELS[x],
            index=0,
            help=(
                "How edges between electrodes are built in every graph:\n"
                "- Functional (PLV): edges reflect phase synchronization between channels "
                "(data-driven, varies per time window). The threshold above is applied to PLV values.\n"
                "- Topological (10-20): edges reflect the physical distance between electrodes "
                "on the scalp (fixed structure, same for every window). The threshold above is applied "
                "to spatial proximity (1 - distance/max_distance)."
            ),
            key="dataset_connectivity_type",
        )

        # --- Augmentation settings ---
        st.markdown("##### Data Augmentation")
        aug_col1, aug_col2, aug_col3, aug_col4 = st.columns([2, 2, 2, 2])

        with aug_col1:
            overlap = st.slider(
                "Window Overlap",
                min_value=0.0,
                max_value=0.75,
                value=0.0,
                step=0.25,
                help=(
                    "Overlap between consecutive time windows (as a fraction). "
                    "0.0 = no overlap (disjoint windows). "
                    "0.5 = each window shares half its duration with the next. "
                    "More overlap = more graphs per subject (data augmentation). "
                    "Example: 60s recording, 5s windows, 0.5 overlap -> ~23 graphs instead of 12."
                ),
                key="dataset_overlap",
            )

        with aug_col2:
            noise_augmentation = st.checkbox(
                "Gaussian Noise",
                value=False,
                help=(
                    "Add augmented copies of each graph with Gaussian noise on node features. "
                    "This helps the model generalize by seeing slightly perturbed versions of the same data."
                ),
                key="dataset_noise_aug",
            )

        with aug_col3:
            noise_std = st.slider(
                "Noise Intensity",
                min_value=0.005,
                max_value=0.05,
                value=0.01,
                step=0.005,
                format="%.3f",
                disabled=not noise_augmentation,
                help=(
                    "Standard deviation of Gaussian noise, relative to each feature's std. "
                    "0.01 = very subtle noise (1% of feature variation). "
                    "0.05 = more aggressive perturbation."
                ),
                key="dataset_noise_std",
            )

        with aug_col4:
            noise_copies = st.selectbox(
                "Noise Copies",
                options=[1, 2, 3],
                index=0,
                disabled=not noise_augmentation,
                help=(
                    "Number of noisy copies per original graph. "
                    "1 copy = dataset doubles. 2 copies = dataset triples."
                ),
                key="dataset_noise_copies",
            )

        build_dataset = st.button(
            "Build Dataset",
            use_container_width=True,
            type="primary" if dataset_info is None else "secondary",
            help="Process all subjects from participants_train.tsv and generate the graph dataset file.",
        )

    # Use detected mode from existing dataset for training, or from build controls
    active_classification_mode = detected_mode if (detected_mode and not build_dataset) else classification_mode

    return {
        "classification_mode": active_classification_mode if active_classification_mode else classification_mode,
        "window_duration": window_duration,
        "plv_threshold": plv_threshold,
        "apply_filter": apply_filter,
        "apply_ica": apply_ica,
        "overlap": overlap,
        "noise_augmentation": noise_augmentation,
        "noise_std": noise_std,
        "noise_copies": int(noise_copies),
        "build_dataset": build_dataset,
        "dataset_exists": dataset_info is not None,
        "dataset_info": dataset_info,
        "connectivity_type": connectivity_type,
    }


def render_training_controls() -> dict[str, Any]:
    """Renders GNN training hyperparameter controls."""
    st.markdown("### Step 2: GNN Training Configuration")
    st.caption(
        "Configure the model architecture and training hyperparameters, then start training."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        model_name = st.selectbox(
            "Model Architecture",
            get_supported_models(),
            index=0,
            help=(
                "Select the Graph Neural Network architecture:\n"
                "- GCN (Graph Convolutional Network): Uses neighborhood averaging with learned weights. "
                "Formula: H^(l+1) = sigma(D_tilde^(-1/2) A_tilde D_tilde^(-1/2) H^(l) W^(l))\n"
                "- GAT (Graph Attention Network): Uses attention mechanisms to weight neighbor importance. "
                "Formula: h_i' = sigma(sum(alpha_ij * W * h_j)), where alpha_ij are learned attention coefficients."
            ),
            key="train_model_name",
        )

    with col2:
        hidden_channels = st.selectbox(
            "Hidden Channels",
            options=[16, 32, 48, 64, 128],
            index=2,
            help=(
                "Number of hidden features per node in graph convolution layers. "
                "Higher values increase model capacity but also increase risk of overfitting and compute time. "
                "Typical range for EEG graphs: 32-128."
            ),
            key="train_hidden_channels",
        )

    with col3:
        dropout = st.slider(
            "Dropout Rate",
            min_value=0.0,
            max_value=0.8,
            value=0.5,
            step=0.05,
            help=(
                "Probability of zeroing each neuron during training (regularization). "
                "Dropout = 0.5 means each neuron has a 50% chance of being deactivated per forward pass. "
                "Helps prevent overfitting. Set to 0.0 to disable."
            ),
            key="train_dropout",
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        epochs = st.number_input(
            "Training Epochs",
            min_value=5,
            max_value=500,
            value=100,
            step=5,
            help=(
                "Number of full passes through the training dataset. "
                "One epoch = all training batches processed once. "
                "More epochs allow the model to learn better, but too many may cause overfitting."
            ),
            key="train_epochs",
        )

    with col5:
        learning_rate = st.selectbox(
            "Learning Rate",
            options=[0.01, 0.005, 0.001, 0.0005, 0.0001],
            index=2,
            help=(
                "Step size for the Adam optimizer gradient updates. "
                "Controls how much model weights change per step: w_new = w_old - lr * dLoss/dw. "
                "Common range: 0.001-0.0001. Higher = faster but unstable; lower = slower but stable."
            ),
            key="train_lr",
        )

    with col6:
        batch_size = st.selectbox(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            index=2,
            help=(
                "Number of graphs processed together in one forward/backward pass. "
                "Larger batches give more stable gradients but require more memory. "
                "Smaller batches add noise that can help generalization."
            ),
            key="train_batch_size",
        )

    col7, col8, col9 = st.columns(3)

    with col7:
        train_ratio = st.slider(
            "Train Split Ratio",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help=(
                "Fraction of subjects used for training (rest goes to validation). "
                "Split is done at the subject level - all graphs from one subject go to the same set. "
                "This prevents data leakage between train and validation."
            ),
            key="train_ratio",
        )

    with col8:
        save_mode = st.selectbox(
            "Checkpoint Save Mode",
            options=["best", "last", "both"],
            index=0,
            help=(
                "When to save model checkpoints:\n"
                "- best: Save only when validation accuracy improves (recommended)\n"
                "- last: Save only the final model after all epochs\n"
                "- both: Save both best and last checkpoints"
            ),
            key="train_save_mode",
        )

    with col9:
        checkpoint_name = st.text_input(
            "Checkpoint Name",
            value="experiment_01",
            help=(
                "Base filename for saved checkpoint(s). "
                "The system appends '_best.pt' or '_last.pt' automatically. "
                "Use descriptive names like 'gcn_binary_v2' to track experiments."
            ),
            key="train_checkpoint_name",
        )

    gat_heads_first_layer = 4
    gat_heads_second_layer = 1

    if model_name == "GAT":
        st.markdown("#### GAT Attention Heads")
        col10, col11 = st.columns(2)

        with col10:
            gat_heads_first_layer = st.selectbox(
                "Heads - First Layer",
                options=[1, 2, 4, 8],
                index=2,
                help=(
                    "Number of parallel attention heads in the first GAT layer. "
                    "Multi-head attention allows the model to attend to neighbors from different representation subspaces. "
                    "Output features are concatenated: output_dim = heads * hidden_channels."
                ),
                key="train_gat_heads_1",
            )

        with col11:
            gat_heads_second_layer = st.selectbox(
                "Heads - Second Layer",
                options=[1, 2, 4],
                index=0,
                help=(
                    "Number of attention heads in the second GAT layer. "
                    "Typically 1 head is used in the final layer (outputs are averaged, not concatenated)."
                ),
                key="train_gat_heads_2",
            )

    # Advanced training settings
    with st.expander("Advanced Training Settings", expanded=False):
        st.caption(
            "These parameters control regularization and learning rate scheduling. "
            "The defaults work well for most cases."
        )
        adv_col1, adv_col2, adv_col3 = st.columns(3)

        with adv_col1:
            weight_decay = st.select_slider(
                "Weight Decay (L2)",
                options=[0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                value=5e-4,
                help=(
                    "L2 regularization strength applied to model weights. "
                    "Penalizes large weights to prevent overfitting: Loss_total = Loss + λ × ||W||². "
                    "Higher values = stronger regularization. Recommended: 1e-4 to 5e-4 for small datasets."
                ),
                key="train_weight_decay",
            )

        with adv_col2:
            scheduler_patience = st.number_input(
                "Scheduler Patience",
                min_value=3,
                max_value=50,
                value=15,
                step=1,
                help=(
                    "Number of epochs to wait without improvement before reducing the learning rate. "
                    "If validation accuracy does not improve for this many epochs, the learning rate is reduced. "
                    "Higher patience = more exploration before reducing LR."
                ),
                key="train_scheduler_patience",
            )

        with adv_col3:
            scheduler_factor = st.select_slider(
                "Scheduler Factor",
                options=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8],
                value=0.5,
                help=(
                    "Factor by which the learning rate is multiplied when patience is exhausted. "
                    "new_lr = old_lr × factor. "
                    "Example: factor=0.5 means LR is halved. Lower factor = more aggressive reduction."
                ),
                key="train_scheduler_factor",
            )

    st.markdown("---")

    start_training = st.button(
        "Start Training",
        type="primary",
        use_container_width=True,
    )

    return {
        "model_name": model_name,
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "train_ratio": float(train_ratio),
        "save_mode": save_mode,
        "checkpoint_name": checkpoint_name.strip() or "experiment_01",
        "gat_heads_first_layer": int(gat_heads_first_layer),
        "gat_heads_second_layer": int(gat_heads_second_layer),
        "weight_decay": float(weight_decay),
        "scheduler_patience": int(scheduler_patience),
        "scheduler_factor": float(scheduler_factor),
        "start_training": start_training,
    }


# ─────────────────────────────────────────────────────────────
# TAB 4: Diagnosis Controls
# ─────────────────────────────────────────────────────────────

def render_diagnosis_controls(checkpoint_dir: str, holdout_dir: str) -> dict[str, Any]:
    """Renders diagnosis/inference controls."""
    st.markdown("#### Diagnosis Settings")
    st.caption(
        "Run the trained GNN model on an EEG recording to predict the diagnosis. "
        "You can test on holdout subjects (reserved data) or upload a new EEG file."
    )

    source_type = st.radio(
        "EEG Source",
        options=["holdout", "upload"],
        format_func=lambda x: "Project Holdout Sample" if x == "holdout" else "Manual Upload (.set file)",
        horizontal=True,
        help=(
            "Choose the source of the EEG recording for diagnosis:\n"
            "- Holdout: Use pre-existing recordings in data/holdout_diagnosis/ that were NOT used during training.\n"
            "- Upload: Upload your own EEGLAB .set file for analysis."
        ),
        key="diag_source_type",
    )

    selected_holdout_subject = None
    uploaded_file = None

    if source_type == "holdout":
        holdout_subjects = get_available_holdout_subjects(holdout_dir)
        if holdout_subjects:
            selected_holdout_subject = st.selectbox(
                "Select Holdout Subject",
                holdout_subjects,
                index=0,
                help="Choose a holdout subject whose recording was reserved for testing (not seen during training).",
                key="diag_holdout_subject",
            )
        else:
            st.warning("No holdout files available in data/holdout_diagnosis/.")
    else:
        uploaded_file = st.file_uploader(
            "Upload EEG file (.set)",
            type=["set"],
            accept_multiple_files=False,
            help=(
                "Upload an EEGLAB .set file. Note: some .set files depend on an associated .fdt file. "
                "For initial tests, the holdout option is more reliable."
            ),
            key="diag_upload",
        )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        available_checkpoints = get_available_checkpoints(checkpoint_dir)

        if available_checkpoints:
            selected_checkpoint = st.selectbox(
                "Model Checkpoint",
                available_checkpoints,
                index=0,
                help=(
                    "Select a trained model checkpoint (.pt file) from the checkpoints/ folder. "
                    "The filename usually encodes the model type and classification mode, e.g. 'gcn_binary_best.pt'."
                ),
                key="diag_checkpoint",
            )
        else:
            selected_checkpoint = None
            st.warning("No checkpoints available in checkpoints/.")

    with col2:
        window_duration = st.slider(
            "Time Window Duration (s)",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help=(
                "Duration of each time window for segmentation during inference. "
                "Should match the window duration used during dataset generation and training for consistent results."
            ),
            key="diag_window_duration",
        )

    with col3:
        plv_threshold = st.slider(
            "PLV Threshold",
            min_value=0.1,
            max_value=0.99,
            value=0.9,
            step=0.05,
            help=(
                "PLV threshold for constructing the brain graph during inference. "
                "Should match the threshold used during training for consistent results."
            ),
            key="diag_plv_threshold",
        )

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        apply_filter = st.checkbox(
            "Band-pass Filter",
            value=True,
            help="Apply FIR band-pass filter (0.5-45 Hz) before inference.",
            key="diag_filter",
        )

    with col5:
        apply_ica = st.checkbox(
            "ICA Cleaning",
            value=False,
            help="Apply ICA artifact removal before inference.",
            key="diag_ica",
        )

    with col6:
        inference_batch_size = st.selectbox(
            "Batch Size",
            options=[4, 8, 16, 32, 64],
            index=2,
            help="Number of graphs to process simultaneously during inference. Does not affect results, only speed.",
            key="diag_batch_size",
        )

    with col7:
        show_window_table = st.checkbox(
            "Show Detailed Table",
            value=True,
            help="Display the per-window prediction table with individual class probabilities for each time window.",
            key="diag_show_table",
        )

    connectivity_choice = st.selectbox(
        "Connectivity Type",
        options=["auto", "functional", "topological"],
        index=0,
        format_func=lambda x: (
            "Auto (use the type the model was trained with)"
            if x == "auto"
            else CONNECTIVITY_TYPE_LABELS[x]
        ),
        help=(
            "How graph edges are built during inference:\n"
            "- Auto: use the connectivity type stored in the selected checkpoint "
            "(recommended — it must match how the model was trained).\n"
            "- Functional (PLV): force phase-synchronization edges.\n"
            "- Topological (10-20): force fixed spatial-proximity edges.\n\n"
            "Forcing a type different from the one used in training will usually degrade accuracy."
        ),
        key="diag_connectivity_type",
    )
    connectivity_type = None if connectivity_choice == "auto" else connectivity_choice

    st.markdown("---")

    run_diagnosis = st.button(
        "Run Diagnosis",
        type="primary",
        use_container_width=True,
    )

    return {
        "source_type": source_type,
        "selected_holdout_subject": selected_holdout_subject,
        "uploaded_file": uploaded_file,
        "selected_checkpoint": selected_checkpoint,
        "window_duration": window_duration,
        "plv_threshold": plv_threshold,
        "apply_filter": apply_filter,
        "apply_ica": apply_ica,
        "inference_batch_size": int(inference_batch_size),
        "show_window_table": show_window_table,
        "run_diagnosis": run_diagnosis,
        "connectivity_type": connectivity_type,
    }


# ─────────────────────────────────────────────────────────────
# Training Progress Rendering
# ─────────────────────────────────────────────────────────────

def render_training_placeholders() -> dict[str, Any]:
    progress_bar = st.empty()
    status_text = st.empty()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_loss = st.empty()
    with col2:
        metric_acc = st.empty()
    with col3:
        metric_precision = st.empty()
    with col4:
        metric_f1 = st.empty()

    chart_placeholder = st.empty()
    summary_placeholder = st.empty()

    return {
        "progress_bar": progress_bar,
        "status_text": status_text,
        "metric_loss": metric_loss,
        "metric_acc": metric_acc,
        "metric_precision": metric_precision,
        "metric_f1": metric_f1,
        "chart_placeholder": chart_placeholder,
        "summary_placeholder": summary_placeholder,
    }


def render_training_epoch_metrics(epoch_info: dict[str, Any], total_epochs: int, placeholders: dict[str, Any]):
    current_epoch = epoch_info["epoch"]
    progress_percent = int((current_epoch / total_epochs) * 100)

    placeholders["progress_bar"].progress(progress_percent)

    lr_str = f" | LR: {epoch_info.get('learning_rate', '?'):.1e}" if "learning_rate" in epoch_info else ""
    placeholders["status_text"].info(
        f"Training in progress... Epoch {current_epoch}/{total_epochs}{lr_str}"
    )

    placeholders["metric_loss"].metric(
        "Train Loss",
        f"{epoch_info['train_loss']:.4f}",
    )
    placeholders["metric_acc"].metric(
        "Val Accuracy",
        f"{epoch_info['val_accuracy'] * 100:.2f}%",
        delta=f"Train: {epoch_info.get('train_accuracy', 0) * 100:.1f}%" if "train_accuracy" in epoch_info else None,
    )
    placeholders["metric_precision"].metric(
        "Val Macro Precision",
        f"{epoch_info['val_macro_precision'] * 100:.2f}%",
    )
    placeholders["metric_f1"].metric(
        "Val Macro F1",
        f"{epoch_info['val_macro_f1'] * 100:.2f}%",
    )


def render_training_history_chart(history_rows: list[dict[str, Any]], chart_placeholder):
    if not history_rows:
        return

    history_df = pd.DataFrame(history_rows)
    columns_to_show = ["train_loss", "val_accuracy", "val_macro_f1"]
    if "val_loss" in history_df.columns:
        columns_to_show.insert(1, "val_loss")
    if "train_accuracy" in history_df.columns:
        columns_to_show.insert(2, "train_accuracy")

    available = [c for c in columns_to_show if c in history_df.columns]
    chart_df = history_df.set_index("epoch")[available]
    chart_placeholder.line_chart(chart_df)


def render_training_final_summary(final_summary: dict[str, Any], summary_placeholder):
    if not final_summary:
        return

    best_metrics = final_summary.get("best_metrics") or {}
    training_report = final_summary.get("training_report") or {}

    summary_placeholder.success(
        (
            f"Training complete! "
            f"Best epoch: {final_summary.get('best_epoch', '-')}, "
            f"Best validation accuracy: {final_summary.get('best_val_accuracy', 0.0) * 100:.2f}%"
        )
    )

    # Quick stats row
    if training_report:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Time", f"{training_report.get('total_time_seconds', 0):.0f}s")
        with col2:
            st.metric("Model Params", f"{training_report.get('model_parameters', {}).get('trainable', 0):,}")
        with col3:
            st.metric("Final LR", f"{training_report.get('final_learning_rate', 0):.1e}")
        with col4:
            gap = training_report.get("overfitting_gap", 0)
            st.metric("Overfit Gap", f"{gap:.4f}", help="Val loss - Train loss. High values indicate overfitting.")

    # Convergence analysis
    if training_report.get("convergence_analysis"):
        analysis = training_report["convergence_analysis"]
        with st.expander("Convergence Analysis and Recommendations", expanded=True):
            if "performance_assessment" in analysis:
                st.info(analysis["performance_assessment"])
            if "plateau_note" in analysis:
                st.warning(analysis["plateau_note"])
            if "overfitting_note" in analysis:
                st.error(analysis["overfitting_note"])
            if "early_peak_note" in analysis:
                st.warning(analysis["early_peak_note"])
            if "late_peak_note" in analysis:
                st.info(analysis["late_peak_note"])

            st.markdown("**Numerical indicators:**")
            st.write(f"- Loss std (last 10%): {analysis.get('loss_std_last_10pct', 'N/A')}")
            st.write(f"- Val loss std (last 10%): {analysis.get('val_loss_std_last_10pct', 'N/A')}")
            st.write(f"- Plateau detected: {analysis.get('plateau_detected', 'N/A')}")
            st.write(f"- Overfitting detected: {analysis.get('overfitting_detected', 'N/A')}")

    # LR schedule changes
    if training_report.get("learning_rate_changes"):
        with st.expander("Learning Rate Schedule Changes", expanded=False):
            for change in training_report["learning_rate_changes"]:
                st.write(
                    f"Epoch {change['epoch']}: {change['old_lr']:.1e} -> {change['new_lr']:.1e}"
                )

    # Detailed JSON report
    with st.expander("Full Training Report (JSON)", expanded=False):
        st.json(
            {
                "best_epoch": final_summary.get("best_epoch"),
                "best_val_accuracy": final_summary.get("best_val_accuracy"),
                "best_metrics": best_metrics,
                "training_report": training_report,
                "config": final_summary.get("config"),
            }
        )


# ─────────────────────────────────────────────────────────────
# Diagnosis Results Rendering
# ─────────────────────────────────────────────────────────────

def render_diagnosis_summary(summary: dict[str, Any]):
    st.subheader("Final Diagnosis Result")

    final_label = summary["final_label"]
    final_confidence = summary["final_confidence"] * 100

    if "Alzheimer" in final_label or "Frontotemporal" in final_label:
        st.error(
            f"**Predicted class:** {final_label} | **Model confidence:** {final_confidence:.2f}%"
        )
    else:
        st.success(
            f"**Predicted class:** {final_label} | **Model confidence:** {final_confidence:.2f}%"
        )

    st.markdown("##### Mean Class Probabilities")
    prob_cols = st.columns(max(1, len(summary["mean_probabilities"])))
    for idx, (class_name, prob_value) in enumerate(summary["mean_probabilities"].items()):
        with prob_cols[idx]:
            st.metric(class_name, f"{prob_value * 100:.2f}%")

    st.markdown("##### Time Window Vote Distribution")
    count_cols = st.columns(max(1, len(summary["window_counts"])))
    for idx, (class_name, count_value) in enumerate(summary["window_counts"].items()):
        ratio = summary["window_ratios"][class_name] * 100
        with count_cols[idx]:
            st.metric(class_name, f"{count_value}/{summary['num_windows']}", f"{ratio:.1f}%")

    st.caption(
        "This result is generated by an experimental AI model and does NOT constitute a clinical diagnosis. "
        "Always consult a qualified medical professional."
    )


def render_diagnosis_performance_metrics(
    window_results: list[dict[str, Any]],
    summary: dict[str, Any],
    true_label: Optional[str] = None,
):
    """
    If true_label is provided, computes real classification metrics.
    If not, shows model consistency metrics.
    """
    if not window_results or not summary:
        return

    import math

    num_windows = summary["num_windows"]
    window_counts = summary["window_counts"]
    class_names = list(window_counts.keys())
    predicted_label = summary["final_label"]

    if true_label and true_label in class_names:
        st.markdown("##### Classification Performance Metrics")
        st.caption(
            f"True label for this subject: **{true_label}**. "
            "Metrics computed by comparing the model's predictions against the known clinical diagnosis."
        )

        # --- Accuracy: proportion of windows that correctly predict the true label ---
        correct_windows = window_counts.get(true_label, 0)
        incorrect_windows = num_windows - correct_windows
        accuracy = correct_windows / num_windows if num_windows > 0 else 0.0

        # --- Average Confidence: mean probability the model assigns to the TRUE class ---
        confidences_true_class = []
        for wr in window_results:
            probs = wr.get("class_probabilities", {})
            if true_label in probs:
                confidences_true_class.append(probs[true_label])
        avg_confidence = (
            sum(confidences_true_class) / len(confidences_true_class)
            if confidences_true_class
            else 0.0
        )

        # --- Precision: among windows predicting the final label, how many are correct? ---
        predicted_count = window_counts.get(predicted_label, 0)
        if predicted_label == true_label:
            # All windows that voted for the predicted (=true) label are correct
            precision = 1.0 if predicted_count > 0 else 0.0
        else:
            # The model's final answer is wrong → precision for that decision = 0
            precision = 0.0

        # --- Recall (= Accuracy here): proportion of windows that identify the true class ---
        recall = accuracy

        # --- F1-Score: harmonic mean of precision and recall ---
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Window Accuracy",
                f"{accuracy * 100:.1f}%",
                help=(
                    "What proportion of time windows predicted the correct diagnosis.\n\n"
                    "**Formula:** Correct Windows ÷ Total Windows\n\n"
                    f"Here: {correct_windows} ÷ {num_windows} = {accuracy * 100:.1f}%"
                ),
            )
        with col2:
            st.metric(
                "Avg. Confidence",
                f"{avg_confidence * 100:.1f}%",
                help=(
                    "The average probability (certainty) that the model assigns to the "
                    "**true** class across all time windows.\n\n"
                    "A high value means the model is consistently confident about the correct answer, "
                    "even in windows where it might not choose it as the top prediction.\n\n"
                    "**Formula:** Mean of P(true class) across all windows."
                ),
            )
        with col3:
            st.metric(
                "Correct / Total",
                f"{correct_windows} / {num_windows}",
                help=(
                    "Raw count of how many time windows agree with the true diagnosis "
                    "vs. the total number of windows analyzed.\n\n"
                    f"{correct_windows} windows predicted '{true_label}' correctly out of {num_windows} total."
                ),
            )
        with col4:
            st.metric(
                "F1-Score",
                f"{f1 * 100:.1f}%",
                help=(
                    "A balanced measure that combines Precision and Recall into a single number.\n\n"
                    "• **Precision** = Is the model's final answer correct? (1.0 if yes, 0.0 if no)\n"
                    "• **Recall** = What fraction of windows support the true diagnosis?\n\n"
                    "**Formula:** F1 = 2 × Precision × Recall ÷ (Precision + Recall)\n\n"
                    f"Here: 2 × {precision:.2f} × {recall:.2f} ÷ ({precision:.2f} + {recall:.2f}) = {f1 * 100:.1f}%"
                ),
            )

        if predicted_label == true_label:
            st.success(
                f"**Diagnosis CORRECT** — The model predicted **'{predicted_label}'**, "
                f"which matches the true clinical label **'{true_label}'**."
            )
        else:
            st.error(
                f"**Diagnosis INCORRECT** — The model predicted **'{predicted_label}'**, "
                f"but the true clinical label is **'{true_label}'**."
            )

        # Explanatory info box
        with st.expander("How are these metrics calculated?", expanded=False):
            st.markdown(
                f"""
**Context:** The EEG recording is divided into **{num_windows} overlapping time windows**.
Each window is independently classified by the Graph Neural Network. The metrics above
summarize how well the model performed across all windows for this subject.

| Metric | Meaning | Formula |
|--------|---------|---------|
| **Window Accuracy** | Fraction of windows that predicted the correct class | Correct Windows ÷ Total Windows |
| **Avg. Confidence** | How certain the model is about the true class (on average) | Mean probability assigned to the true class |
| **Correct / Total** | Simple count of correct vs. total windows | — |
| **F1-Score** | Balanced score combining precision and recall | 2 × Precision × Recall ÷ (Precision + Recall) |

**Why these metrics?**
- *Window Accuracy* tells you how consistently the model identifies the correct diagnosis across different segments of the EEG.
- *Average Confidence* reveals whether the model is "sure" about its answer or "guessing" — even a correct prediction with low confidence is less reliable.
- *F1-Score* penalizes the model if its final decision is wrong (precision = 0), providing a stricter evaluation.

**Interpretation guide:**
- **> 80%** accuracy + **> 70%** confidence → Strong, reliable prediction
- **50–80%** accuracy → The model is uncertain; consider additional clinical evaluation
- **< 50%** accuracy → The model struggles with this recording; the result should not be trusted
"""
            )

    else:
        st.markdown("##### Model Consistency Metrics")
        st.caption(
            "No ground truth available for this subject. The metrics below indicate "
            "how consistent the model's predictions are across different time segments of the EEG."
        )

        agree_count = window_counts.get(predicted_label, 0)
        consistency = agree_count / num_windows if num_windows > 0 else 0.0

        entropy = 0.0
        for cls in class_names:
            p = window_counts.get(cls, 0) / num_windows if num_windows > 0 else 0
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(class_names)) if len(class_names) > 1 else 1.0
        agreement_score = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Consistency",
                f"{consistency * 100:.1f}%",
                help=(
                    "What percentage of time windows agree with the final prediction.\n\n"
                    "100% means every window produced the same answer — the model is very consistent.\n\n"
                    f"**Formula:** Windows voting '{predicted_label}' ÷ Total Windows = "
                    f"{agree_count} ÷ {num_windows} = {consistency * 100:.1f}%"
                ),
            )
        with col2:
            st.metric(
                "Agreement Score",
                f"{agreement_score * 100:.1f}%",
                help=(
                    "An entropy-based measure of how concentrated the predictions are.\n\n"
                    "• 100% = all windows agree on one class (zero entropy)\n"
                    "• 0% = windows are spread equally across all classes (maximum entropy)\n\n"
                    "**Formula:** 1 − (Entropy ÷ Max Possible Entropy)"
                ),
            )
        with col3:
            st.metric(
                "Majority Windows",
                f"{agree_count}/{num_windows}",
                help=(
                    "Number of windows that voted for the winning class vs. total windows.\n\n"
                    f"{agree_count} out of {num_windows} windows predicted '{predicted_label}'."
                ),
            )

        with st.expander("How are these metrics calculated?", expanded=False):
            st.markdown(
                f"""
**Context:** Since no verified clinical diagnosis is available for this subject,
we cannot measure "correctness." Instead, we measure **consistency** — how much
the model agrees with itself across the {num_windows} time windows.

| Metric | Meaning | Formula |
|--------|---------|---------|
| **Consistency** | % of windows agreeing with the final answer | Majority votes ÷ Total windows |
| **Agreement Score** | Entropy-based concentration measure | 1 − (Shannon Entropy ÷ Max Entropy) |
| **Majority Windows** | Raw vote count for the winning class | — |

**Interpretation guide:**
- **> 90%** consistency → The model is very confident and stable across the recording
- **70–90%** → Moderate agreement; the prediction is likely but not certain
- **< 70%** → Significant disagreement between windows; interpret with caution

**What is entropy?** Entropy measures disorder. If all windows say the same thing,
entropy is 0 (perfect order). If windows are evenly split between classes, entropy is
at its maximum (complete disorder). The Agreement Score converts this into an
intuitive 0–100% scale.
"""
            )


def render_diagnosis_charts(
    window_results: list[dict[str, Any]],
    summary: dict[str, Any],
):
    """Renders interactive charts summarizing the diagnosis results."""
    if not window_results or not summary:
        return

    st.markdown("##### Visual Summary")

    chart_col1, chart_col2 = st.columns(2)

    # ─── Donut Chart: Window Vote Distribution ───
    with chart_col1:
        window_counts = summary["window_counts"]
        labels = list(window_counts.keys())
        values = list(window_counts.values())

        color_map = {
            "Healthy Control": "#81c784",
            "Alzheimer's Disease": "#e57373",
            "Frontotemporal Dementia": "#ffb74d",
        }
        colors = [color_map.get(lbl, "#b39ddb") for lbl in labels]

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,
                    marker=dict(colors=colors),
                    textinfo="label+percent",
                    textposition="outside",
                    hovertemplate="<b>%{label}</b><br>Windows: %{value}<br>Ratio: %{percent}<extra></extra>",
                )
            ]
        )
        fig_donut.update_layout(
            title=dict(text="Window Vote Distribution", x=0.5, font=dict(size=14)),
            showlegend=False,
            height=350,
            margin=dict(t=60, b=20, l=20, r=20),
            annotations=[
                dict(
                    text=f"{summary['num_windows']}<br>windows",
                    x=0.5, y=0.5, font_size=14, showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ─── Bar Chart: Mean Class Probabilities ───
    with chart_col2:
        mean_probs = summary["mean_probabilities"]
        prob_labels = list(mean_probs.keys())
        prob_values = [v * 100 for v in mean_probs.values()]
        colors_bar = [color_map.get(lbl, "#b39ddb") for lbl in prob_labels]

        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=prob_labels,
                    y=prob_values,
                    marker_color=colors_bar,
                    text=[f"{v:.1f}%" for v in prob_values],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Mean Probability: %{y:.2f}%<extra></extra>",
                )
            ]
        )
        fig_bar.update_layout(
            title=dict(text="Mean Class Probabilities", x=0.5, font=dict(size=14)),
            yaxis=dict(title="Probability (%)", range=[0, 100]),
            xaxis=dict(title=""),
            height=350,
            margin=dict(t=60, b=20, l=50, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ─── Confidence Timeline ───
    st.markdown("##### Prediction Confidence Over Time")

    window_indices = [wr["window_index"] for wr in window_results]
    confidences = [wr["confidence"] * 100 for wr in window_results]
    pred_labels = [wr["predicted_label"] for wr in window_results]

    color_sequence = [color_map.get(lbl, "#b39ddb") for lbl in pred_labels]

    fig_timeline = go.Figure()

    # Add confidence line
    fig_timeline.add_trace(
        go.Scatter(
            x=window_indices,
            y=confidences,
            mode="lines+markers",
            line=dict(color="#7e57c2", width=2),
            marker=dict(
                size=8,
                color=color_sequence,
                line=dict(width=1, color="#4a148c"),
            ),
            hovertemplate=(
                "<b>Window %{x}</b><br>"
                "Confidence: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
            name="Confidence",
        )
    )

    # Add a horizontal line for average confidence
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    fig_timeline.add_hline(
        y=avg_conf,
        line_dash="dash",
        line_color="#9575cd",
        annotation_text=f"Avg: {avg_conf:.1f}%",
        annotation_position="top right",
    )

    fig_timeline.update_layout(
        xaxis=dict(title="Time Window Index", dtick=1),
        yaxis=dict(title="Confidence (%)", range=[0, 105]),
        height=300,
        margin=dict(t=30, b=40, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ─── Legend for marker colors ───
    legend_parts = []
    for lbl, clr in color_map.items():
        if lbl in [wr["predicted_label"] for wr in window_results]:
            legend_parts.append(
                f'<span style="color:{clr}; font-size:20px;">●</span> {lbl}'
            )
    if legend_parts:
        st.markdown(
            "&nbsp;&nbsp;&nbsp;".join(legend_parts),
            unsafe_allow_html=True,
        )
    st.caption(
        "Each dot represents one time window. The color indicates the predicted class for that window. "
        "The dashed line shows the average confidence across all windows."
    )


def render_window_results_table(window_results: list[dict[str, Any]], true_label: Optional[str] = None):
    if not window_results:
        st.warning("No per-window results to display.")
        return

    rows = []

    for row in window_results:
        flat_row = {
            "Window #": row["window_index"],
            "Start (s)": round(float(row["start_sec"]), 2),
            "End (s)": round(float(row["end_sec"]), 2),
            "Prediction": row["predicted_label"],
            "Confidence": round(float(row["confidence"]), 4),
        }

        # Add per-window ground truth metrics if true_label is available
        if true_label:
            flat_row["Correct"] = "Yes" if row["predicted_label"] == true_label else "No"
            probs = row.get("class_probabilities", {})
            flat_row["P(True Class)"] = round(float(probs.get(true_label, 0.0)), 4)

        for class_name, prob in row["class_probabilities"].items():
            flat_row[f"P({class_name})"] = round(float(prob), 4)

        rows.append(flat_row)

    df = pd.DataFrame(rows)

    st.markdown("##### Per-Window Predictions")

    if true_label:
        correct_count = sum(1 for r in rows if r["Correct"] == "Yes")
        st.caption(
            f"Ground truth: **{true_label}** | "
            f"Correct windows: {correct_count}/{len(rows)} | "
            f"The 'Correct' column indicates whether each window's prediction matches the true diagnosis. "
            f"'P(True Class)' shows the probability assigned to the true class by the model."
        )

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_checkpoint_info(checkpoint_info: dict[str, Any]):
    if not checkpoint_info:
        return

    with st.expander("Model / Checkpoint Details", expanded=False):
        st.json(checkpoint_info)

