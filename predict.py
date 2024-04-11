import pandas as pd
import numpy as np
import pickle
from scipy.fft import fft
import scipy.stats
from scipy.signal import find_peaks

def extract_features(data_matrix):
    # Convert data_matrix to DataFrame if it's a numpy array
    if isinstance(data_matrix, np.ndarray):
        data_matrix = pd.DataFrame(data_matrix)
    
    features_list = []

    for index, row in data_matrix.iterrows():
        row_array = row.to_numpy()

        # Basic statistical features
        mean_val = np.mean(row_array)
        std_val = np.std(row_array)
        max_val = np.max(row_array)
        min_val = np.min(row_array)
        median_val = np.median(row_array)
        var_val = np.var(row_array)
        range_val = max_val - min_val
        skew_val = scipy.stats.skew(row_array)
        kurtosis_val = scipy.stats.kurtosis(row_array)

        # Percentile features
        pct_10 = np.percentile(row_array, 10)
        pct_25 = np.percentile(row_array, 25)
        pct_75 = np.percentile(row_array, 75)
        pct_90 = np.percentile(row_array, 90)

        # Rate of change features
        diff = np.diff(row_array)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        min_diff = np.min(diff)

        # Window-based features
        first_half_mean = np.mean(row_array[:len(row_array)//2])
        second_half_mean = np.mean(row_array[len(row_array)//2:])
        first_half_std = np.std(row_array[:len(row_array)//2])
        second_half_std = np.std(row_array[len(row_array)//2:])

        # Peak analysis
        peaks, _ = find_peaks(row_array)
        peak_count = len(peaks)
        peak_mean_amplitude = np.mean(row_array[peaks]) if peaks.size > 0 else 0

        # Rolling features
        rolling_mean = pd.Series(row_array).rolling(window=5).mean().iloc[-1]
        rolling_std = pd.Series(row_array).rolling(window=5).std().iloc[-1]

        # Concatenate all features
        features = [
            mean_val, std_val, max_val, min_val, median_val, var_val, range_val,
            skew_val, kurtosis_val, pct_10, pct_25, pct_75, pct_90,
            mean_diff, max_diff, min_diff, first_half_mean, second_half_mean,
            first_half_std, second_half_std, peak_count, peak_mean_amplitude,
            rolling_mean, rolling_std
        ]

        features_list.append(features)

    return pd.DataFrame(features_list, columns=[f'feature_{i+1}' for i in range(24)])

def main():
    # Load the trained model, scaler, and sampler from pickle
    with open('model_scaler_sampler.pkl', 'rb') as f:
        model, scaler, sampler = pickle.load(f)

    test_data = pd.read_csv('test.csv', header=None)  # Assuming test.csv is correctly formatted as N x 24 matrix
    test_features = extract_features(test_data)

    # Scaling and sampling test data features
    test_features_scaled = scaler.transform(test_features)
    test_features_resampled, _ = sampler.fit_resample(test_features_scaled, np.zeros(test_features_scaled.shape[0]))  # labels not needed for prediction

    # Make predictions
    predictions = model.predict(test_features_resampled)

    # Save results
    pd.DataFrame(predictions).to_csv('Result.csv', index=False, header=False)

if __name__ == "__main__":
    main()