from __future__ import annotations

import os
import sys
import json
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ui_components import (
    setup_page,
    render_signal_controls,
    render_graph_controls,
    render_dataset_controls,
    render_training_controls,
    render_diagnosis_controls,
    render_training_placeholders,
    render_training_epoch_metrics,
    render_training_history_chart,
    render_training_final_summary,
    render_diagnosis_summary,
    render_diagnosis_performance_metrics,
    render_diagnosis_charts,
    render_window_results_table,
    render_checkpoint_info,
    CONNECTIVITY_TYPE_LABELS,
)

from src.preprocessing import load_and_preprocess_eeg
from src.graph_builder import get_connectivity_for_ui
from src.dataset_builder import build_pyg_dataset
from src.training import train_gnn
from src.inference import run_recording_inference


import pandas as pd

APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DERIVATIVES_DIR = os.path.join(DATA_DIR, "derivatives")
HOLDOUT_DIR = os.path.join(DATA_DIR, "holdout_diagnosis")

PARTICIPANTS_TSV = os.path.join(DATA_DIR, "participants_train.tsv")
PARTICIPANTS_HOLDOUT_TSV = os.path.join(DATA_DIR, "participants_holdout.tsv")
DATASET_PATH = os.path.join(DATA_DIR, "eeg_graphs_dataset.pt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

GROUP_TO_CLASS_LABEL = {
    "C": "Healthy Control",
    "A": "Alzheimer's Disease",
    "F": "Frontotemporal Dementia",
}


def get_holdout_true_label(subject_id: str) -> str | None:
    """Look up the true class label for a holdout subject from participants_holdout.tsv."""
    if not os.path.exists(PARTICIPANTS_HOLDOUT_TSV):
        return None
    try:
        df = pd.read_csv(PARTICIPANTS_HOLDOUT_TSV, sep="\t")
        row = df[df["participant_id"] == subject_id]
        if row.empty:
            return None
        group = row.iloc[0]["Group"]
        return GROUP_TO_CLASS_LABEL.get(group)
    except Exception:
        return None


@st.cache_resource
def get_eeg_data(subject_id: str, apply_filter: bool, apply_ica: bool):
    if not subject_id:
        return None, None

    file_path = os.path.join(
        DERIVATIVES_DIR,
        subject_id,
        "eeg",
        f"{subject_id}_task-eyesclosed_eeg.set",
    )

    if not os.path.exists(file_path):
        return None, file_path

    raw_signal = load_and_preprocess_eeg(
        file_path=file_path,
        apply_filter=apply_filter,
        apply_ica=apply_ica,
    )
    return raw_signal, file_path


def build_networkx_graph_from_plv(plv_matrix, channel_names, threshold: float):
    num_channels = len(channel_names)
    G = nx.Graph()

    for i, ch in enumerate(channel_names):
        G.add_node(i, label=ch)

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            weight = float(plv_matrix[i, j])
            if weight >= threshold:
                G.add_edge(i, j, weight=weight)

    return G


def draw_brain_graph(G, threshold: float, scale: float = 1.0, connectivity_type: str = "functional"):
    base_w, base_h = 8, 6
    fig_graph, ax = plt.subplots(figsize=(base_w * scale, base_h * scale))
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="#b39ddb",
        node_size=int(600 * scale),
        edgecolors="#4a148c",
    )

    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        ax=ax,
        font_size=10,
        font_weight="bold",
    )

    edges = list(G.edges(data=True))
    weights = [max(1.0, edge[2]["weight"] * 4) for edge in edges]

    if len(edges) > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=weights,
            edge_color="#9575cd",
            alpha=0.6,
        )

    if connectivity_type == "topological":
        title_metric = f"Spatial Proximity Threshold: {threshold}"
        title_kind = "Topological (10-20 layout)"
    else:
        title_metric = f"PLV Threshold: {threshold}"
        title_kind = "Functional (PLV)"

    ax.set_title(
        f"2D Brain Connectivity Network — {title_kind}\n({title_metric})",
        fontsize=16,
        pad=20,
    )
    ax.axis("off")

    return fig_graph


def plot_window_signal(window_data, channel_names, sfreq, window_index, prediction_label):
    num_channels, num_samples = window_data.shape
    time_axis = [i / sfreq for i in range(num_samples)]

    fig, ax = plt.subplots(figsize=(12, 8))

    max_amp = max(float(abs(window_data).max()), 1e-6)
    spacing = max_amp * 3.0

    for ch_idx in range(num_channels):
        offset = (num_channels - ch_idx - 1) * spacing
        ax.plot(time_axis, window_data[ch_idx] + offset, linewidth=0.8)
        ax.text(
            time_axis[0] - 0.02 * max(time_axis[-1], 1e-6),
            offset,
            channel_names[ch_idx],
            fontsize=8,
            va="center",
        )

    ax.set_title(
        f"Time Window {window_index} Signal View | Predicted: {prediction_label}",
        fontsize=14,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.set_yticks([])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    return fig


def save_diagnosis_report(
    inference_result: dict,
    diagnosis_controls: dict,
    true_label: str | None,
) -> str:
    """Save a JSON report summarizing the diagnosis run to the results/ directory.

    The report captures every detail needed to fully reproduce and interpret the
    diagnosis, including the connectivity type (functional vs. topological) that
    was actually used to build the graph edges.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def _round(value, ndigits=6):
        """Round numbers safely, leaving non-numeric values (e.g. None) untouched."""
        if isinstance(value, (int, float)):
            return round(value, ndigits)
        return value

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if diagnosis_controls["source_type"] == "holdout":
        source_id = diagnosis_controls["selected_holdout_subject"] or "unknown"
    else:
        uploaded_file = diagnosis_controls.get("uploaded_file")
        source_id = (
            os.path.splitext(uploaded_file.name)[0]
            if uploaded_file is not None and getattr(uploaded_file, "name", None)
            else "uploaded_file"
        )

    filename = f"diagnosis_{source_id}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    summary = inference_result["summary"]
    checkpoint_info = inference_result["checkpoint_info"]
    window_results = inference_result["window_results"]

    # --- Connectivity type resolution ----------------------------------------
    # The user's UI choice (None == "auto"); what the model was actually run with.
    requested_connectivity = diagnosis_controls.get("connectivity_type")
    resolved_connectivity = checkpoint_info.get("connectivity_type")
    connectivity_label = CONNECTIVITY_TYPE_LABELS.get(
        resolved_connectivity, resolved_connectivity
    )
    # The threshold means different things depending on connectivity type.
    threshold_kind = (
        "proximity_threshold"
        if resolved_connectivity == "topological"
        else "plv_threshold"
    )

    # Class label mapping used by the model (int id -> human-readable name).
    class_labels = checkpoint_info.get("class_labels", {})
    class_labels_serializable = {str(k): v for k, v in class_labels.items()}

    report = {
        "timestamp": datetime.now().isoformat(),
        "source": {
            "type": diagnosis_controls["source_type"],
            "subject_id": source_id,
            "true_label": true_label,
        },
        "recording": {
            "sampling_frequency_hz": _round(inference_result.get("sfreq"), 2),
            "num_channels": len(inference_result.get("channel_names", []) or []),
            "channel_names": inference_result.get("channel_names"),
            "num_windows": summary["num_windows"],
        },
        "inference_parameters": {
            "checkpoint": diagnosis_controls["selected_checkpoint"],
            "window_duration_sec": diagnosis_controls["window_duration"],
            threshold_kind: diagnosis_controls["plv_threshold"],
            "apply_filter": diagnosis_controls["apply_filter"],
            "apply_ica": diagnosis_controls["apply_ica"],
            "batch_size": diagnosis_controls["inference_batch_size"],
            "connectivity_type": resolved_connectivity,
            "connectivity_type_label": connectivity_label,
            "connectivity_type_requested": (
                "auto" if requested_connectivity is None else requested_connectivity
            ),
        },
        "model": {
            "model_name": checkpoint_info["model_name"],
            "hidden_channels": checkpoint_info["config"].get("hidden_channels"),
            "dropout": checkpoint_info["config"].get("dropout"),
            "num_node_features": checkpoint_info["config"].get("num_node_features"),
            "num_classes": checkpoint_info["config"].get("num_classes"),
            "classification_mode": checkpoint_info["config"].get("classification_mode"),
            "trained_connectivity_type": checkpoint_info["config"].get(
                "connectivity_type", "functional"
            ),
            "class_labels": class_labels_serializable,
            "training_epoch": checkpoint_info.get("epoch"),
            "training_metrics": checkpoint_info.get("metrics"),
            "gat_heads_first_layer": checkpoint_info["config"].get("gat_heads_first_layer"),
            "gat_heads_second_layer": checkpoint_info["config"].get("gat_heads_second_layer"),
        },
        "results": {
            "final_label": summary["final_label"],
            "final_confidence": _round(summary["final_confidence"]),
            "num_windows": summary["num_windows"],
            "connectivity_type": resolved_connectivity,
            "mean_probabilities": {
                k: _round(v) for k, v in summary["mean_probabilities"].items()
            },
            "window_vote_counts": summary["window_counts"],
            "window_vote_ratios": {
                k: _round(v) for k, v in summary["window_ratios"].items()
            },
            "correct": (summary["final_label"] == true_label) if true_label else None,
        },
        "per_window": [
            {
                "window_index": w["window_index"],
                "start_sec": _round(w["start_sec"], 2),
                "end_sec": _round(w["end_sec"], 2),
                "predicted_label": w["predicted_label"],
                "confidence": _round(w["confidence"]),
                "class_probabilities": {
                    k: _round(v) for k, v in w["class_probabilities"].items()
                },
            }
            for w in window_results
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return filepath


# ─────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────

setup_page()

if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None

if "diagnosis_ran" not in st.session_state:
    st.session_state.diagnosis_ran = False

if "diagnosis_source_signature" not in st.session_state:
    st.session_state.diagnosis_source_signature = None

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='text-align: center;'> EEG Analysis using GNN for Dementia Detection</h1>
    <p style='text-align: center; color: gray;'>
        Interactive interface for EEG signal visualization, functional connectivity graph extraction,
        GNN training, and AI-assisted diagnosis.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# ─────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "EEG Signals",
        "Brain Graph",
        "Model Training",
        "Diagnosis",
    ]
)

# ─────────────────────────────────────────────────────────────
# TAB 1: EEG Signal Exploration
# ─────────────────────────────────────────────────────────────

with tab1:
    st.header("EEG Signal Exploration")
    st.caption(
        "Visualize preprocessed EEG recordings. Select a subject and configure display settings below."
    )

    signal_settings = render_signal_controls(DATA_DIR)

    selected_subject = signal_settings["selected_subject"]
    window_duration = signal_settings["window_duration"]
    apply_filter = signal_settings["apply_filter"]
    apply_ica = signal_settings["apply_ica"]

    if selected_subject is None:
        st.warning("No subject is currently available. Check your data/derivatives/ folder.")
    else:
        with st.spinner("Loading and preprocessing EEG data..."):
            raw_signal, file_path = get_eeg_data(
                selected_subject,
                apply_filter=apply_filter,
                apply_ica=apply_ica,
            )

        if raw_signal is None:
            st.error(
                f"File not found at expected path: {file_path}. "
                "Please check your data folder structure."
            )
        else:
            st.success(
                f"Data loaded for **{selected_subject}** — "
                f"{raw_signal.info['sfreq']} Hz sampling rate, {len(raw_signal.ch_names)} channels."
            )

            st.subheader(f"Preprocessed EEG Segment ({window_duration}s)")

            eeg_scale = st.slider(
                "Figure scale",
                min_value=0.5,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Adjust the size of the EEG signal plot.",
                key="eeg_signal_scale",
            )

            fig = raw_signal.plot(
                duration=window_duration,
                n_channels=min(15, len(raw_signal.ch_names)),
                show=False,
                scalings="auto",
                title=f"EEG Traces — {selected_subject}",
            )
            fig.set_size_inches(10 * eeg_scale, 6 * eeg_scale)
            st.pyplot(fig)

            st.info(
                f"Subject: {selected_subject} | "
                f"Filter: {'ON' if apply_filter else 'OFF'} | "
                f"ICA: {'ON' if apply_ica else 'OFF'} | "
                f"Channels: {len(raw_signal.ch_names)}"
            )

# ─────────────────────────────────────────────────────────────
# TAB 2: Brain Graph
# ─────────────────────────────────────────────────────────────

with tab2:
    st.header("Brain Connectivity Graph")
    st.caption(
        "Build the connectivity graph between EEG channels and visualize it. "
        "Choose between functional connectivity (PLV synchronization) and topological "
        "connectivity (physical 10-20 electrode layout)."
    )

    graph_settings = render_graph_controls(DATA_DIR)

    graph_subject = graph_settings["selected_subject"]
    graph_window = graph_settings["window_duration"]
    graph_plv = graph_settings["plv_threshold"]
    graph_filter = graph_settings["apply_filter"]
    graph_ica = graph_settings["apply_ica"]
    graph_connectivity = graph_settings["connectivity_type"]

    if graph_subject is None:
        st.warning("No subject available for graph computation.")
    else:
        with st.spinner("Loading EEG data for graph computation..."):
            raw_signal_graph, _ = get_eeg_data(
                graph_subject,
                apply_filter=graph_filter,
                apply_ica=graph_ica,
            )

        if raw_signal_graph is None:
            st.warning("EEG data could not be loaded for the selected subject.")
        else:
            spinner_msg = (
                "Computing spatial proximity matrix and rendering brain graph..."
                if graph_connectivity == "topological"
                else "Computing PLV matrix and rendering brain graph..."
            )
            with st.spinner(spinner_msg):
                channel_names = raw_signal_graph.ch_names
                plv_matrix = get_connectivity_for_ui(
                    raw_signal_graph,
                    duration_sec=graph_window,
                    connectivity_type=graph_connectivity,
                )

                G = build_networkx_graph_from_plv(
                    plv_matrix=plv_matrix,
                    channel_names=channel_names,
                    threshold=graph_plv,
                )

                graph_scale = st.slider(
                    "Figure scale",
                    min_value=0.5,
                    max_value=2.0,
                    value=0.8,
                    step=0.1,
                    help="Adjust the size of the brain graph plot.",
                    key="brain_graph_scale",
                )

                fig_graph = draw_brain_graph(
                    G,
                    threshold=graph_plv,
                    scale=graph_scale,
                    connectivity_type=graph_connectivity,
                )
                st.pyplot(fig_graph)

                metric_name = (
                    "spatial proximity" if graph_connectivity == "topological" else "PLV"
                )
                st.info(
                    f"Connectivity: **{graph_connectivity}** | "
                    f"Graph Statistics: **{G.number_of_nodes()} nodes** (electrodes) and "
                    f"**{G.number_of_edges()} edges** (connections with {metric_name} >= {graph_plv:.2f})."
                )

# ─────────────────────────────────────────────────────────────
# TAB 3: Model Training
# ─────────────────────────────────────────────────────────────

with tab3:
    st.header("Model Training")
    st.caption(
        "Configure dataset generation and train a Graph Neural Network for EEG-based dementia classification."
    )

    dataset_controls = render_dataset_controls(DATASET_PATH)
    classification_mode = dataset_controls["classification_mode"]

    if dataset_controls["build_dataset"]:
        with st.spinner("Building dataset from raw EEG files... This may take several minutes."):
            try:
                dataset = build_pyg_dataset(
                    tsv_path=PARTICIPANTS_TSV,
                    base_data_dir=DATA_DIR,
                    output_path=DATASET_PATH,
                    threshold=dataset_controls["plv_threshold"],
                    window_duration=dataset_controls["window_duration"],
                    apply_filter=dataset_controls["apply_filter"],
                    apply_ica=dataset_controls["apply_ica"],
                    classification_mode=classification_mode,
                    overlap=dataset_controls["overlap"],
                    noise_augmentation=dataset_controls["noise_augmentation"],
                    noise_std=dataset_controls["noise_std"],
                    noise_copies=dataset_controls["noise_copies"],
                    connectivity_type=dataset_controls["connectivity_type"],
                )
                st.success(
                    f"Dataset built successfully! Saved to `data/eeg_graphs_dataset.pt` "
                    f"with **{len(dataset)} graphs**."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"An error occurred while building the dataset: {exc}")

    st.markdown("---")

    training_controls = render_training_controls()
    training_placeholders = render_training_placeholders()

    # Show what mode will be used for training
    if dataset_controls["dataset_info"]:
        ds_info = dataset_controls["dataset_info"]
        st.info(
            f"Training will use the existing dataset: "
            f"**{ds_info['num_graphs']} graphs**, "
            f"**{ds_info['num_classes']} classes** ({classification_mode}) | "
            f"Path: `{DATASET_PATH}`"
        )
    else:
        st.warning("No dataset available. Build a dataset first before training.")

    if training_controls["start_training"]:
        if not os.path.exists(DATASET_PATH):
            st.error("Dataset not found. Please click 'Build / Update Dataset' first.")
        else:
            history_rows = []

            try:
                for info in train_gnn(
                    epochs=training_controls["epochs"],
                    learning_rate=training_controls["learning_rate"],
                    batch_size=training_controls["batch_size"],
                    dataset_path=DATASET_PATH,
                    model_name=training_controls["model_name"],
                    hidden_channels=training_controls["hidden_channels"],
                    dropout=training_controls["dropout"],
                    train_ratio=training_controls["train_ratio"],
                    save_mode=training_controls["save_mode"],
                    checkpoint_dir=CHECKPOINT_DIR,
                    checkpoint_name=training_controls["checkpoint_name"],
                    num_node_features=4,
                    gat_heads_first_layer=training_controls["gat_heads_first_layer"],
                    gat_heads_second_layer=training_controls["gat_heads_second_layer"],
                    seed=42,
                    window_duration=dataset_controls["window_duration"],
                    plv_threshold=dataset_controls["plv_threshold"],
                    classification_mode=classification_mode,
                    weight_decay=training_controls["weight_decay"],
                    scheduler_patience=training_controls["scheduler_patience"],
                    scheduler_factor=training_controls["scheduler_factor"],
                    connectivity_type=dataset_controls["connectivity_type"],
                ):
                    if info.get("status") == "completed":
                        training_placeholders["progress_bar"].progress(100)
                        training_placeholders["status_text"].success(
                            "Training completed successfully!"
                        )
                        render_training_final_summary(
                            info,
                            training_placeholders["summary_placeholder"],
                        )
                    else:
                        history_rows.append(info)
                        render_training_epoch_metrics(
                            epoch_info=info,
                            total_epochs=training_controls["epochs"],
                            placeholders=training_placeholders,
                        )
                        render_training_history_chart(
                            history_rows,
                            training_placeholders["chart_placeholder"],
                        )

            except Exception as exc:
                st.error(f"Training failed: {exc}")

    # ─── Checkpoint Management ───
    st.markdown("---")
    st.markdown("### Saved Checkpoints")

    from app.ui_components import get_available_checkpoints

    saved_checkpoints = get_available_checkpoints(CHECKPOINT_DIR)

    if not saved_checkpoints:
        st.caption("No saved checkpoints yet.")
    else:
        st.caption(f"{len(saved_checkpoints)} checkpoint(s) found in `checkpoints/`.")

        col_del1, col_del2 = st.columns([3, 1])

        with col_del1:
            selected_to_delete = st.multiselect(
                "Select checkpoints to delete",
                options=saved_checkpoints,
                default=[],
                help="Select one or more checkpoints to remove from disk.",
                key="checkpoints_to_delete",
            )

        with col_del2:
            st.markdown("")
            st.markdown("")
            select_all = st.checkbox("Select all", key="select_all_checkpoints")

        if select_all:
            selected_to_delete = saved_checkpoints

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            delete_selected = st.button(
                "Delete Selected",
                disabled=len(selected_to_delete) == 0,
                use_container_width=True,
                key="btn_delete_selected_checkpoints",
            )

        with col_btn2:
            delete_all = st.button(
                "Delete All Checkpoints",
                type="primary",
                use_container_width=True,
                key="btn_delete_all_checkpoints",
            )

        if delete_selected and selected_to_delete:
            for ckpt_name in selected_to_delete:
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
            st.success(f"Deleted {len(selected_to_delete)} checkpoint(s).")
            st.rerun()

        if delete_all:
            count = 0
            for ckpt_name in saved_checkpoints:
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    count += 1
            st.success(f"Deleted all {count} checkpoint(s).")
            st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 4: Diagnosis
# ─────────────────────────────────────────────────────────────

with tab4:
    st.header("GNN Model Inference / Diagnosis")
    st.caption(
        "Test the trained model on reserved holdout EEG recordings or upload a new EEG file for AI-assisted diagnosis."
    )

    diagnosis_controls = render_diagnosis_controls(CHECKPOINT_DIR, HOLDOUT_DIR)

    current_source_signature = (
        diagnosis_controls["source_type"],
        diagnosis_controls["selected_holdout_subject"],
        diagnosis_controls["selected_checkpoint"],
        diagnosis_controls["window_duration"],
        diagnosis_controls["plv_threshold"],
        diagnosis_controls["apply_filter"],
        diagnosis_controls["apply_ica"],
        diagnosis_controls["inference_batch_size"],
    )

    if diagnosis_controls["run_diagnosis"]:
        if diagnosis_controls["selected_checkpoint"] is None:
            st.error("Please select a valid checkpoint first.")
        else:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                diagnosis_controls["selected_checkpoint"],
            )

            eeg_path_for_inference = None
            temp_file_created = False

            if diagnosis_controls["source_type"] == "holdout":
                selected_holdout_subject = diagnosis_controls["selected_holdout_subject"]

                if selected_holdout_subject is None:
                    st.error("Please select a holdout subject.")
                else:
                    eeg_path_for_inference = os.path.join(
                        HOLDOUT_DIR,
                        selected_holdout_subject,
                        "eeg",
                        f"{selected_holdout_subject}_task-eyesclosed_eeg.set",
                    )
            else:
                uploaded_file = diagnosis_controls["uploaded_file"]

                if uploaded_file is None:
                    st.error("Please upload an EEG .set file first.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".set") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        eeg_path_for_inference = tmp_file.name
                        temp_file_created = True

            if eeg_path_for_inference is not None:
                try:
                    with st.spinner("Running diagnosis... Preprocessing, graph building, and inference in progress."):
                        inference_result = run_recording_inference(
                            eeg_file_path=eeg_path_for_inference,
                            checkpoint_path=checkpoint_path,
                            window_duration=diagnosis_controls["window_duration"],
                            threshold=diagnosis_controls["plv_threshold"],
                            batch_size=diagnosis_controls["inference_batch_size"],
                            apply_filter=diagnosis_controls["apply_filter"],
                            apply_ica=diagnosis_controls["apply_ica"],
                            connectivity_type=diagnosis_controls["connectivity_type"],
                        )

                    st.session_state.diagnosis_result = inference_result
                    st.session_state.diagnosis_ran = True
                    st.session_state.diagnosis_source_signature = current_source_signature

                    # Save diagnosis report to results/ directory
                    true_label_for_report = None
                    if diagnosis_controls["source_type"] == "holdout" and diagnosis_controls["selected_holdout_subject"]:
                        true_label_for_report = get_holdout_true_label(diagnosis_controls["selected_holdout_subject"])

                    report_path = save_diagnosis_report(
                        inference_result=inference_result,
                        diagnosis_controls=diagnosis_controls,
                        true_label=true_label_for_report,
                    )
                    st.success(f"Diagnosis report saved to: `{os.path.relpath(report_path, PROJECT_ROOT)}`")

                except Exception as exc:
                    st.error(f"Diagnosis failed: {exc}")
                finally:
                    if (
                        temp_file_created
                        and eeg_path_for_inference
                        and os.path.exists(eeg_path_for_inference)
                    ):
                        os.remove(eeg_path_for_inference)

    if (
        st.session_state.diagnosis_ran
        and st.session_state.diagnosis_result is not None
        and st.session_state.diagnosis_source_signature == current_source_signature
    ):
        inference_result = st.session_state.diagnosis_result

        st.markdown("---")
        st.markdown(
            """
#### Understanding the Results

The model analyzes the EEG recording by dividing it into multiple **time windows** —
short segments of brain activity (typically a few seconds each). Each window is independently
processed through a **Graph Neural Network (GNN)**, which converts the electrical signals
from different brain regions into a graph structure and classifies the pattern it observes.

The **final diagnosis** is determined by a majority vote: whichever class receives the most
votes across all time windows becomes the predicted label. The **confidence** reflects the
average probability the model assigns to its chosen class.

Below you will find:
1. **Final Diagnosis Result** — the model's overall prediction and confidence level
2. **Classification Performance Metrics** — quantitative evaluation of the prediction quality
3. **Time Window Signal Inspection** — detailed view of individual EEG segments
"""
        )
        st.markdown("---")

        render_diagnosis_summary(inference_result["summary"])

        # Determine true label for holdout subjects
        true_label = None
        if diagnosis_controls["source_type"] == "holdout" and diagnosis_controls["selected_holdout_subject"]:
            true_label = get_holdout_true_label(diagnosis_controls["selected_holdout_subject"])

        render_diagnosis_performance_metrics(
            inference_result["window_results"],
            inference_result["summary"],
            true_label=true_label,
        )

        render_diagnosis_charts(
            inference_result["window_results"],
            inference_result["summary"],
        )

        render_checkpoint_info(inference_result["checkpoint_info"])

        if diagnosis_controls["show_window_table"]:
            render_window_results_table(
                inference_result["window_results"], true_label=true_label
            )

        st.subheader("Time Window Signal Inspection")
        window_results = inference_result["window_results"]
        windows_array = inference_result["windows_array"]
        channel_names = inference_result["channel_names"]
        sfreq = inference_result["sfreq"]

        if len(window_results) > 0:
            window_options = [
                f"Window {row['window_index']} | {row['predicted_label']} | confidence={row['confidence']:.3f}"
                for row in window_results
            ]

            selected_window_ui = st.selectbox(
                "Select a time window to inspect",
                options=list(range(len(window_options))),
                format_func=lambda idx: window_options[idx],
                key="diagnosis_window_selector",
                help="Choose a specific time window to view its raw EEG signal and the model's prediction for it.",
            )

            selected_window_result = window_results[selected_window_ui]
            selected_window_data = windows_array[selected_window_ui]

            fig_window = plot_window_signal(
                window_data=selected_window_data,
                channel_names=channel_names,
                sfreq=sfreq,
                window_index=selected_window_result["window_index"],
                prediction_label=selected_window_result["predicted_label"],
            )
            st.pyplot(fig_window)

            probs_text = " | ".join(
                [
                    f"{class_name}: {prob * 100:.2f}%"
                    for class_name, prob in selected_window_result["class_probabilities"].items()
                ]
            )

            st.info(
                f"Time Window {selected_window_result['window_index']} | "
                f"Time range: {selected_window_result['start_sec']:.2f}s — "
                f"{selected_window_result['end_sec']:.2f}s | "
                f"{probs_text}"
            )


