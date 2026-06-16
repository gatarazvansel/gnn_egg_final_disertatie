import os
from typing import Optional

import mne
import numpy as np

# Keep MNE logs compact in terminal / Streamlit
mne.set_log_level("ERROR")


STANDARD_EEG_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]


def get_standard_eeg_channels() -> list[str]:
    """
    Returns the target 10-20 EEG channel list used across the project.
    """
    return STANDARD_EEG_CHANNELS.copy()


def load_eeg_file(file_path: str) -> mne.io.BaseRaw:
    """
    Loads an EEGLAB .set file into an MNE Raw object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EEG file not found: {file_path}")

    print(f"-> Loading EEG file: {os.path.basename(file_path)}")
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw


def select_standard_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Keeps only the standard 10-20 channels that are present in the recording.
    """
    available_channels = set(raw.ch_names)
    channels_to_pick = [ch for ch in STANDARD_EEG_CHANNELS if ch in available_channels]

    if not channels_to_pick:
        raise ValueError(
            "None of the standard EEG channels were found in the recording. "
            f"Available channels: {raw.ch_names}"
        )

    raw_selected = raw.copy().pick(channels_to_pick)
    print(f"-> Selected {len(channels_to_pick)} EEG channels: {channels_to_pick}")
    return raw_selected


def apply_bandpass_filter(
    raw: mne.io.BaseRaw,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
) -> mne.io.BaseRaw:
    """
    Applies a band-pass FIR filter to the EEG signal.
    """
    print(f"-> Applying band-pass filter ({l_freq} - {h_freq} Hz)...")
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    return raw_filtered


def apply_ica_cleaning(
    raw: mne.io.BaseRaw,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> mne.io.BaseRaw:
    """
    Applies ICA artifact cleaning.
    """
    n_channels = len(raw.ch_names)

    if n_components is None:
        n_components = min(15, n_channels)

    if n_components < 2:
        raise ValueError("ICA requires at least 2 channels.")

    print(f"-> Running ICA artifact cleaning with {n_components} components...")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw)

    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned)
    return raw_cleaned


def load_and_preprocess_eeg(
    file_path: str,
    apply_filter: bool = True,
    apply_ica: bool = False,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
) -> mne.io.BaseRaw:
    """
    Loads an EEG file and applies the standard preprocessing pipeline:
    1. Load file
    2. Keep only standard channels
    3. Optional band-pass filtering
    4. Optional ICA cleaning
    """
    raw = load_eeg_file(file_path)
    raw = select_standard_channels(raw)

    if apply_filter:
        raw = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)

    if apply_ica:
        raw = apply_ica_cleaning(raw)

    sfreq = raw.info["sfreq"]
    print(f"-> Final sampling rate: {sfreq} Hz")
    print(f"-> Final channel count: {len(raw.ch_names)}")

    return raw


def extract_time_windows(
    raw: mne.io.BaseRaw,
    window_duration: float = 5.0,
    overlap: float = 0.0,
    reject_amplitude_uv: Optional[float] = None,
    detrend: Optional[int] = 1,
) -> mne.Epochs:
    """
    Splits a continuous EEG recording into fixed-length time windows.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed EEG signal.
    window_duration : float
        Time window length in seconds.
    overlap : float
        Time window overlap in seconds.
    reject_amplitude_uv : Optional[float]
        If provided, drops time windows exceeding this amplitude threshold in microvolts.
    detrend : Optional[int]
        Passed to MNE Epochs detrend parameter.
    """
    if window_duration <= 0:
        raise ValueError("window_duration must be > 0.")

    if overlap < 0:
        raise ValueError("overlap must be >= 0.")

    if overlap >= window_duration:
        raise ValueError("overlap must be smaller than window_duration.")

    print(
        f"-> Segmenting signal into {window_duration:.2f}s time windows "
        f"(overlap={overlap:.2f}s)..."
    )

    events = mne.make_fixed_length_events(
        raw,
        duration=window_duration,
        overlap=overlap,
    )

    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=window_duration,
        baseline=None,
        preload=True,
        detrend=detrend,
        verbose=False,
    )

    if reject_amplitude_uv is not None:
        reject_criteria = dict(eeg=reject_amplitude_uv * 1e-6)
        epochs.drop_bad(reject=reject_criteria)

    print(f"-> Extracted {len(epochs)} time windows.")
    return epochs


def time_windows_to_numpy(epochs: mne.Epochs) -> np.ndarray:
    """
    Converts MNE Epochs to a NumPy array of shape:
    (num_windows, num_channels, num_timepoints)
    """
    data = epochs.get_data()
    print(f"-> Time window tensor shape: {data.shape}")
    return data


if __name__ == "__main__":
    test_file = "../data/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

    if not os.path.exists(test_file):
        print(f"Error: Could not find the file at path: {test_file}")
    else:
        print("\n--- PREPROCESSING TEST ---")
        raw_signal = load_and_preprocess_eeg(
            test_file,
            apply_filter=True,
            apply_ica=False,
        )

        mne_epochs = extract_time_windows(
            raw_signal,
            window_duration=5.0,
            overlap=0.0,
            reject_amplitude_uv=None,
        )

        data_array = time_windows_to_numpy(mne_epochs)

        print("\n=== FINAL RESULT ===")
        print(f"Shape: {data_array.shape}")
        print("(Number of Time Windows, Number of Electrodes, Samples per Window)")
