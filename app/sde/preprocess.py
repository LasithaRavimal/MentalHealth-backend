import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

N_AVERAGED = 16
TRIAL_LENGTH = 9216
ELECTRODE_COL_INDEX = [4, 5, 6, 7, 8, 9, 10, 11, 12]

def averaged_by_N_rows(a, n):
    usable_len = (a.shape[0] // n) * n
    a = a[:usable_len]
    return a.reshape(-1, n, a.shape[1]).mean(axis=1)

def preprocess_eeg(csv_path: str):
    df = pd.read_csv(csv_path, header=None)

    if df.shape[0] < TRIAL_LENGTH:
        raise ValueError("EEG file too short")

    X_list = []

    for start in range(0, df.shape[0] - TRIAL_LENGTH + 1, TRIAL_LENGTH):
        trial_df = df.iloc[start:start + TRIAL_LENGTH]
        eeg = trial_df.iloc[:, ELECTRODE_COL_INDEX].values
        eeg_avg = averaged_by_N_rows(eeg, N_AVERAGED)
        eeg_flat = eeg_avg.reshape(-1)
        X_list.append(eeg_flat.astype(np.float32))

    X = np.array(X_list, dtype=np.float32)
    X = normalize(X, axis=1, norm="max")

    return X.reshape(X.shape[0], 9, 576, 1)
