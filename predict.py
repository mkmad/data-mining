import pandas as pd
import numpy as np
import pickle
from scipy.fft import fft
import scipy.stats

def predict_single_sample(model, scaler, sample):
    scaled_sample = scaler.transform([sample])
    return model.predict(scaled_sample)[0]

# Function to extract features
def extract_features(data_matrix):
    features_list = []

    for row_array in data_matrix:
        # row_array = row.to_numpy()

        # Statistical features
        mean_val = np.mean(row_array)
        std_val = np.std(row_array)
        max_val = np.max(row_array)
        min_val = np.min(row_array)
        median_val = np.median(row_array)
        var_val = np.var(row_array)
        range_val = max_val - min_val
        skew_val = scipy.stats.skew(row_array)
        kurtosis_val = scipy.stats.kurtosis(row_array)

        # Frequency-domain features: Calculate FFT and select the first few components as features
        fft_values = fft(row_array)
        fft_magnitude = np.abs(fft_values)
        fft_features = fft_magnitude[:15]  # Make sure this count contributes correctly to the total feature count

        # Concatenate all features
        features = [
            mean_val, std_val, max_val, min_val, median_val, var_val, range_val,
            skew_val, kurtosis_val
        ] + fft_features.tolist()

        # Ensure we have exactly 24 features
        assert len(features) == 24, f"Expected 24 features, but got {len(features)}"
        features_list.append(features)

    return pd.DataFrame(features_list, columns=[f'feature_{i+1}' for i in range(24)])

def main():
    with open('model_scaler.pkl', 'rb') as f:
        model = pickle.load(f)

    test_data = pd.read_csv('test.csv', header=None)  # Assuming test.csv is correctly formatted as N x 24 matrix
    print(test_data)
    test_data_matrix = test_data.to_numpy()

    test_features = extract_features(test_data_matrix)

    # Make predictions

    predictions = model.predict(test_features)

    # Assuming predictions is a 2D array and we need the first column
    prediction_column = predictions[:, 0] if predictions.ndim > 1 else predictions

    # Save results
    pd.DataFrame(prediction_column).to_csv('Result.csv', index=False, header=False)

if __name__ == "__main__":
    main()