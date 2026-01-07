import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize

N_AVERAGED = 16
TRIAL_LENGTH = 9216

BASE_DIR = Path(__file__).resolve().parent
COLUMN_LABELS_PATH = BASE_DIR / "columnLabels.csv"

def averaged_by_N_rows(a, n):
    usable_len = (a.shape[0] // n) * n
    a = a[:usable_len]
    return a.reshape(-1, n, a.shape[1]).mean(axis=1)

def preprocess_eeg(csv_path: str):
    # Load column labels
    column_labels = pd.read_csv(COLUMN_LABELS_PATH).columns

    # Load EEG CSV
    df = pd.read_csv(csv_path, header=None, names=column_labels)

    electrodes = column_labels[4:]

    X_trials = []

    for trial_id in df.trial.unique():
        trial_df = df[df.trial == trial_id]

        if len(trial_df) != TRIAL_LENGTH:
            continue

        eeg = trial_df[electrodes].values
        eeg_avg = averaged_by_N_rows(eeg, N_AVERAGED)
        eeg_flat = eeg_avg.reshape(-1).astype(np.float32)
        X_trials.append(eeg_flat)

    if len(X_trials) == 0:
        raise ValueError("No valid EEG trials found")

    X = np.array(X_trials, dtype=np.float32)

    # Normalize SAME AS TRAINING
    X = normalize(
        X.reshape(-1, len(electrodes)),
        axis=0,
        norm="max"
    ).reshape(X.shape)

    time_steps = X.shape[1] // len(electrodes)
    return X.reshape(X.shape[0], len(electrodes), time_steps, 1)
