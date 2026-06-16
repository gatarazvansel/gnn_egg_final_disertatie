from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

from scipy import signal
from scipy.signal import hilbert
from torch_geometric.data import Data


# Supported connectivity (edge-construction) strategies.
#   - "functional"  : data-driven edges from Phase Locking Value (PLV) in the alpha band.
#   - "topological" : fixed edges from the physical 10-20 electrode layout (spatial proximity).
CONNECTIVITY_TYPES = ("functional", "topological")
DEFAULT_CONNECTIVITY_TYPE = "functional"

# Cache for the standard 10-20 electrode coordinates so we only query MNE once.
_MONTAGE_POSITIONS_CACHE: dict[tuple[str, ...], np.ndarray] = {}


def normalize_connectivity_type(connectivity_type: str | None) -> str:
    """Validates and normalizes the connectivity type string."""
    if connectivity_type is None:
        return DEFAULT_CONNECTIVITY_TYPE

    value = str(connectivity_type).strip().lower()
    if value not in CONNECTIVITY_TYPES:
        raise ValueError(
            f"Unsupported connectivity_type '{connectivity_type}'. "
            f"Supported types: {CONNECTIVITY_TYPES}"
        )
    return value


def get_electrode_positions(channel_names: list[str]) -> np.ndarray:
    """
    Returns the 3D coordinates (in meters) of the given EEG channels using the
    standard 10-20 montage provided by MNE.

    Parameters
    ----------
    channel_names : list[str]
        EEG channel names (e.g. ['Fp1', 'Fp2', ...]).

    Returns
    -------
    np.ndarray
        Shape: (num_channels, 3). Channels not found in the montage get NaN.
    """
    cache_key = tuple(channel_names)
    if cache_key in _MONTAGE_POSITIONS_CACHE:
        return _MONTAGE_POSITIONS_CACHE[cache_key]

    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]

    positions = np.full((len(channel_names), 3), np.nan, dtype=np.float64)
    for idx, ch in enumerate(channel_names):
        coords = pos_dict.get(ch)
        if coords is None:
            # Try a case-insensitive fallback match.
            for name, value in pos_dict.items():
                if name.lower() == ch.lower():
                    coords = value
                    break
        if coords is not None:
            positions[idx] = np.asarray(coords, dtype=np.float64)

    _MONTAGE_POSITIONS_CACHE[cache_key] = positions
    return positions


def compute_topological_matrix(channel_names: list[str]) -> np.ndarray:
    """
    Computes a topological (spatial) connectivity matrix based on the physical
    distances between electrodes in the standard 10-20 system.

    The matrix encodes spatial proximity in [0, 1]:
        proximity(i, j) = 1 - dist(i, j) / max_dist
    so that the closest electrode pair is near 1 and the farthest is near 0.
    This matrix is static (it does not depend on the EEG signal) and can be
    thresholded exactly like the PLV matrix.

    Parameters
    ----------
    channel_names : list[str]
        EEG channel names.

    Returns
    -------
    np.ndarray
        Shape: (num_channels, num_channels), zero diagonal.
    """
    positions = get_electrode_positions(channel_names)
    num_channels = positions.shape[0]

    proximity_matrix = np.zeros((num_channels, num_channels), dtype=np.float32)

    # Pairwise Euclidean distances; NaN positions yield NaN distances (no edge).
    distances = np.full((num_channels, num_channels), np.nan, dtype=np.float64)
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            diff = positions[i] - positions[j]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))
            distances[j, i] = distances[i, j]

    finite_distances = distances[np.isfinite(distances)]
    if finite_distances.size == 0:
        # No valid coordinates -> empty (disconnected) graph.
        return proximity_matrix

    max_dist = float(finite_distances.max())
    if max_dist <= 0:
        return proximity_matrix

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            dist = distances[i, j]
            if not np.isfinite(dist):
                continue
            proximity = 1.0 - (dist / max_dist)
            proximity = max(0.0, min(1.0, proximity))
            proximity_matrix[i, j] = proximity
            proximity_matrix[j, i] = proximity

    np.fill_diagonal(proximity_matrix, 0.0)
    return proximity_matrix


def bandpass_filter_window(
    window_data: np.ndarray,
    sfreq: float,
    low_freq: float = 8.0,
    high_freq: float = 12.0,
    order: int = 4,
) -> np.ndarray:
    """
    Applies a Butterworth band-pass filter to one EEG time window.

    Parameters
    ----------
    window_data : np.ndarray
        Shape: (num_channels, num_samples)
    sfreq : float
        Sampling frequency.
    low_freq : float
        Lower cutoff frequency.
    high_freq : float
        Upper cutoff frequency.
    order : int
        Filter order.
    """
    if window_data.ndim != 2:
        raise ValueError(
            f"window_data must have shape (num_channels, num_samples), got {window_data.shape}"
        )

    nyquist = sfreq / 2.0
    if high_freq >= nyquist:
        raise ValueError(
            f"high_freq ({high_freq}) must be lower than Nyquist frequency ({nyquist})."
        )

    b, a = signal.butter(
        order,
        [low_freq, high_freq],
        btype="bandpass",
        fs=sfreq,
    )
    filtered_data = signal.filtfilt(b, a, window_data, axis=1)
    return filtered_data


def compute_plv_matrix(
    window_data: np.ndarray,
    sfreq: float,
    plv_band: tuple[float, float] = (8.0, 12.0),
) -> np.ndarray:
    """
    Computes the Phase Locking Value (PLV) matrix for one EEG time window.

    Parameters
    ----------
    window_data : np.ndarray
        Shape: (num_channels, num_samples)
    sfreq : float
        Sampling frequency.
    plv_band : tuple[float, float]
        Frequency band used before Hilbert transform.
    """
    low_freq, high_freq = plv_band
    filtered_data = bandpass_filter_window(
        window_data=window_data,
        sfreq=sfreq,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    num_channels = filtered_data.shape[0]

    analytic_signal = hilbert(filtered_data, axis=1)
    instantaneous_phase = np.angle(analytic_signal)

    plv_matrix = np.zeros((num_channels, num_channels), dtype=np.float32)

    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            phase_diff = instantaneous_phase[i, :] - instantaneous_phase[j, :]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))

            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv

    np.fill_diagonal(plv_matrix, 0.0)
    return plv_matrix


def threshold_adjacency_matrix(
    matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Thresholds a weighted adjacency matrix while preserving weights above threshold.
    """
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")

    thresholded = np.where(matrix >= threshold, matrix, 0.0).astype(np.float32)
    np.fill_diagonal(thresholded, 0.0)
    return thresholded


def build_adjacency_matrix(
    window_data: np.ndarray,
    threshold: float = 0.5,
    sfreq: float = 500.0,
    plv_band: tuple[float, float] = (8.0, 12.0),
    connectivity_type: str = DEFAULT_CONNECTIVITY_TYPE,
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """
    Computes the weighted adjacency matrix for one EEG time window and applies thresholding.

    The edges are built using one of two strategies:
    - "functional"  : Phase Locking Value (PLV) synchronization between channels.
    - "topological" : fixed spatial proximity from the 10-20 electrode layout.

    Returns
    -------
    np.ndarray
        Weighted adjacency matrix of shape (num_channels, num_channels).
        Values below threshold are set to 0.
    """
    connectivity_type = normalize_connectivity_type(connectivity_type)

    if connectivity_type == "topological":
        if channel_names is None:
            raise ValueError(
                "channel_names is required for topological connectivity."
            )
        weighted_matrix = compute_topological_matrix(channel_names=channel_names)
    else:
        weighted_matrix = compute_plv_matrix(
            window_data=window_data,
            sfreq=sfreq,
            plv_band=plv_band,
        )

    adj_matrix = threshold_adjacency_matrix(weighted_matrix, threshold=threshold)
    return adj_matrix


def extract_node_features(
    window_data: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Computes node features for one EEG time window using mean PSD power in 4 bands:
    Delta, Theta, Alpha, Beta.

    Parameters
    ----------
    window_data : np.ndarray
        Shape: (num_channels, num_samples)
    sfreq : float
        Sampling frequency.

    Returns
    -------
    np.ndarray
        Shape: (num_channels, 4)
    """
    if window_data.ndim != 2:
        raise ValueError(
            f"window_data must have shape (num_channels, num_samples), got {window_data.shape}"
        )

    bands = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 12.0),
        "Beta": (12.0, 30.0),
    }

    num_channels = window_data.shape[0]
    node_features = np.zeros((num_channels, 4), dtype=np.float32)

    # Convert to microvolts for easier PSD scaling
    window_data_uv = window_data * 1e6

    for ch in range(num_channels):
        freqs, psd = signal.welch(
            window_data_uv[ch, :],
            fs=sfreq,
            nperseg=min(int(sfreq * 2), window_data_uv.shape[1]),
        )

        for i, (_, (fmin, fmax)) in enumerate(bands.items()):
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)

            if np.any(idx_band):
                band_power = np.mean(psd[idx_band])
            else:
                band_power = 0.0

            node_features[ch, i] = band_power

    return node_features


def adjacency_to_edge_index_and_attr(
    adj_matrix: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a weighted adjacency matrix to PyG edge_index and edge_attr.

    Returns
    -------
    edge_index : torch.Tensor
        Shape: (2, num_edges)
    edge_attr : torch.Tensor
        Shape: (num_edges, 1)
    """
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(
            f"adj_matrix must be square, got shape {adj_matrix.shape}"
        )

    src, dst = np.nonzero(adj_matrix)
    weights = adj_matrix[src, dst].astype(np.float32)

    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long,
    )

    edge_attr = torch.tensor(
        weights,
        dtype=torch.float32,
    ).view(-1, 1)

    return edge_index, edge_attr


def build_pyg_graph_from_window(
    window_data: np.ndarray,
    sfreq: float,
    threshold: float,
    label: int | None = None,
    subject_id: str | None = None,
    window_index: int | None = None,
    channel_names: list[str] | None = None,
    connectivity_type: str = DEFAULT_CONNECTIVITY_TYPE,
) -> Data:
    """
    Builds a single PyTorch Geometric graph from one EEG time window.

    Graph content
    -------------
    x         : node features (band powers)
    edge_index: graph connectivity
    edge_attr : weighted edge values (PLV or spatial proximity)
    y         : optional label
    subject_id: optional metadata
    window_index: optional metadata
    channel_names: optional metadata
    connectivity_type: "functional" (PLV) or "topological" (10-20 layout)
    """
    connectivity_type = normalize_connectivity_type(connectivity_type)

    x_np = extract_node_features(window_data=window_data, sfreq=sfreq)
    adj_matrix = build_adjacency_matrix(
        window_data=window_data,
        threshold=threshold,
        sfreq=sfreq,
        connectivity_type=connectivity_type,
        channel_names=channel_names,
    )
    edge_index, edge_attr = adjacency_to_edge_index_and_attr(adj_matrix)

    graph_data = Data(
        x=torch.tensor(x_np, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    if label is not None:
        graph_data.y = torch.tensor([label], dtype=torch.long)

    if subject_id is not None:
        graph_data.subject_id = subject_id

    if window_index is not None:
        graph_data.window_index = int(window_index)

    if channel_names is not None:
        graph_data.channel_names = channel_names

    return graph_data


def visualize_brain_graph(
    adj_matrix: np.ndarray,
    channel_names: list[str],
    title: str = "Functional Connectivity (EEG Graph PLV)",
):
    """
    Visualizes a weighted functional connectivity graph using NetworkX.
    """
    if adj_matrix.shape[0] != len(channel_names):
        raise ValueError(
            "adj_matrix size and channel_names length do not match."
        )

    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: name for i, name in enumerate(channel_names)}
    G = nx.relabel_nodes(G, mapping)

    plt.figure(figsize=(10, 8))
    pos = nx.circular_layout(G)

    edges = G.edges(data=True)
    weights = [max(1.0, abs(data["weight"]) * 5) for _, _, data in edges]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=700,
        node_color="lightblue",
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=11,
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=weights,
        edge_color="gray",
        alpha=0.7,
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_plv_for_ui(
    raw_signal,
    duration_sec: float,
) -> np.ndarray:
    """
    Helper function for Streamlit:
    extracts the first `duration_sec` seconds from the raw signal and computes
    the unthresholded PLV matrix.

    Thresholding is meant to be applied later in the UI.
    """
    sfreq = raw_signal.info["sfreq"]
    max_samples = int(duration_sec * sfreq)

    data, _ = raw_signal[:, :max_samples]

    plv_matrix = build_adjacency_matrix(
        window_data=data,
        threshold=0.0,
        sfreq=sfreq,
    )
    return plv_matrix


def get_connectivity_for_ui(
    raw_signal,
    duration_sec: float,
    connectivity_type: str = DEFAULT_CONNECTIVITY_TYPE,
) -> np.ndarray:
    """
    Helper function for Streamlit:
    returns the unthresholded weighted connectivity matrix for the selected
    connectivity strategy (PLV or topological). Thresholding is applied later
    in the UI.
    """
    connectivity_type = normalize_connectivity_type(connectivity_type)
    sfreq = raw_signal.info["sfreq"]
    max_samples = int(duration_sec * sfreq)

    data, _ = raw_signal[:, :max_samples]

    return build_adjacency_matrix(
        window_data=data,
        threshold=0.0,
        sfreq=sfreq,
        connectivity_type=connectivity_type,
        channel_names=list(raw_signal.ch_names),
    )


def build_graphs_from_time_windows(
    windows_array: np.ndarray,
    sfreq: float,
    threshold: float,
    label: int | None = None,
    subject_id: str | None = None,
    channel_names: list[str] | None = None,
    connectivity_type: str = DEFAULT_CONNECTIVITY_TYPE,
) -> list[Data]:
    """
    Builds a list of PyG graphs from an array of EEG time windows.

    Parameters
    ----------
    windows_array : np.ndarray
        Shape: (num_windows, num_channels, num_samples)
    sfreq : float
        Sampling frequency.
    threshold : float
        Edge threshold (PLV or spatial proximity).
    label : int | None
        Optional graph label.
    subject_id : str | None
        Optional subject identifier.
    channel_names : list[str] | None
        Optional channel names (required for topological connectivity).
    connectivity_type : str
        "functional" (PLV) or "topological" (10-20 electrode layout).

    Returns
    -------
    list[Data]
        One PyG graph per time window.
    """
    connectivity_type = normalize_connectivity_type(connectivity_type)

    if windows_array.ndim != 3:
        raise ValueError(
            f"windows_array must have shape (num_windows, num_channels, num_samples), got {windows_array.shape}"
        )

    graphs = []

    for window_idx, window_data in enumerate(windows_array):
        graph = build_pyg_graph_from_window(
            window_data=window_data,
            sfreq=sfreq,
            threshold=threshold,
            label=label,
            subject_id=subject_id,
            window_index=window_idx,
            channel_names=channel_names,
            connectivity_type=connectivity_type,
        )
        graphs.append(graph)

    return graphs


if __name__ == "__main__":
    import os
    from src.preprocessing import load_and_preprocess_eeg, extract_time_windows, time_windows_to_numpy

    test_file = "../data/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

    if not os.path.exists(test_file):
        print(f"File not found: {test_file}")
    else:
        print("\n--- GRAPH BUILDER TEST ---")

        raw_signal = load_and_preprocess_eeg(
            test_file,
            apply_filter=True,
            apply_ica=False,
        )

        mne_epochs = extract_time_windows(raw_signal, window_duration=5.0)
        windows_array = time_windows_to_numpy(mne_epochs)

        test_window = windows_array[0]
        channel_names = raw_signal.ch_names
        sfreq = raw_signal.info["sfreq"]

        features = extract_node_features(test_window, sfreq=sfreq)
        print("\n=== NODE FEATURES ===")
        print(f"Shape: {features.shape}")

        adj_matrix = build_adjacency_matrix(
            test_window,
            threshold=0.8,
            sfreq=sfreq,
        )
        num_connections = np.count_nonzero(adj_matrix) // 2
        print(f"Strong connections (PLV): {num_connections}")

        edge_index, edge_attr = adjacency_to_edge_index_and_attr(adj_matrix)
        print(f"edge_index shape: {edge_index.shape}")
        print(f"edge_attr shape: {edge_attr.shape}")

        pyg_graph = build_pyg_graph_from_window(
            window_data=test_window,
            sfreq=sfreq,
            threshold=0.8,
            label=1,
            subject_id="sub-001",
            window_index=0,
            channel_names=channel_names,
        )
        print("\n=== PYG GRAPH ===")
        print(pyg_graph)

        visualize_brain_graph(adj_matrix, channel_names)


