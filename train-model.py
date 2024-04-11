import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fft import fft
import scipy.stats
from scipy.signal import find_peaks

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Function to read the CSV files
def read_csv_files(insulin_file, cgm_file):
    insulin_data = pd.read_csv(insulin_file, parse_dates=['Date'], usecols=['Date', 'BWZ Carb Input (grams)'])
    cgm_data = pd.read_csv(cgm_file, parse_dates=['Date'], usecols=['Date', 'Sensor Glucose (mg/dL)'])
    return insulin_data, cgm_data

# Function to extract meal data
def extract_meal_data(insulin_data, cgm_data):
    meal_data_matrix = []
    insulin_sorted = insulin_data.dropna(subset=['BWZ Carb Input (grams)'])
    insulin_sorted = insulin_sorted[insulin_data['BWZ Carb Input (grams)'] > 0]
    insulin_sorted = insulin_sorted.sort_values('Date')

    for index, row in insulin_sorted.iterrows():
        tm = row['Date']

        # Condition a
        next_meal_time = insulin_sorted[insulin_sorted['Date'] > tm]
        if not next_meal_time.empty and next_meal_time.iloc[0]['Date'] < tm + timedelta(hours=2):
            # Condition b
            tp = next_meal_time.iloc[0]['Date']
            start_time = tp - timedelta(minutes=30)
            end_time = tp + timedelta(hours=2)
        elif not next_meal_time.empty and next_meal_time.iloc[0]['Date'] == tm + timedelta(hours=2):
            # Condition c
            start_time = tm + timedelta(hours=1, minutes=30)
            end_time = tm + timedelta(hours=4)
        else:
            # No meal from tm to tm+2 hours
            start_time = tm - timedelta(minutes=30)
            end_time = tm + timedelta(hours=2)

        mask = (cgm_data['Date'] >= start_time) & (cgm_data['Date'] <= end_time)
        data_slice = cgm_data.loc[mask]['Sensor Glucose (mg/dL)'].tolist()

        if len(data_slice) >= 24:  # Ensure there are 30 records for 2.5 hours at 5-minute intervals
            meal_data_matrix.append(data_slice[:24])

    return pd.DataFrame(meal_data_matrix)


# Function to extract no meal data
def extract_no_meal_data(insulin_data, cgm_data):
    no_meal_data_matrix = []
    insulin_sorted = insulin_data.sort_values('Date')
    
    # Track the end time of the last meal to start searching for no meal data
    last_meal_end_time = None

    for index in range(len(insulin_sorted) - 1):
        row = insulin_sorted.iloc[index]
        next_row = insulin_sorted.iloc[index + 1]

        if pd.notna(row['BWZ Carb Input (grams)']) and row['BWZ Carb Input (grams)'] > 0:
            last_meal_end_time = row['Date'] + timedelta(hours=2)
        elif last_meal_end_time is not None:
            # Checking the post-absorptive period after 2 hours from the last meal
            if row['Date'] >= last_meal_end_time:
                # Checking for next meal within next 2 hours window
                if next_row['Date'] - row['Date'] >= timedelta(hours=2) or \
                        (next_row['Date'] - row['Date'] < timedelta(hours=2) and next_row['BWZ Carb Input (grams)'] == 0):
                    
                    # Extract CGM data for this no meal stretch
                    start_time = row['Date']
                    end_time = start_time + timedelta(hours=2)
                    mask = (cgm_data['Date'] >= start_time) & (cgm_data['Date'] <= end_time)
                    data_slice = cgm_data.loc[mask]['Sensor Glucose (mg/dL)'].tolist()
                    
                    if len(data_slice) >= 24:  # Ensure the slice has exactly 2 hours of data in 5-minute intervals
                        no_meal_data_matrix.append(data_slice[:24])
                    # Update the last meal end time based on current row's timing
                    last_meal_end_time = end_time

    return pd.DataFrame(no_meal_data_matrix)

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


def train_model(features, meal_features_count, no_meal_features_count, n_splits=5):
    X = features
    y = [1] * meal_features_count + [0] * no_meal_features_count

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Oversampling using SMOTE
    sampler = SMOTE()
    X_res, y_res = sampler.fit_resample(X_scaled, y)

    # Using Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=123)

    # Expanded Hyperparameters to tune
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': [None, 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.01, 0.1, 1.0]
    }

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    grid_search = GridSearchCV(model, param_grid, cv=skf, scoring='f1')  # Consider using F1 score for imbalanced data

    grid_search.fit(X_res, y_res)
    best_model = grid_search.best_estimator_

    print(f"Best CV score: {grid_search.best_score_}")
    print(f"Best model parameters: {grid_search.best_params_}")

    # Retrain the final model on the entire dataset with best parameters
    best_model.fit(X_res, y_res)
    print("Final model trained on the entire dataset for deployment.")

    return best_model, scaler, sampler

# Function to retrieve the trained model
def consolidate_features(insulin_file='InsulinData.csv', cgm_file='CGMData.csv'):
    insulin_data, cgm_data = read_csv_files(insulin_file, cgm_file)
    meal_data_matrix = extract_meal_data(insulin_data, cgm_data)
    no_meal_data_matrix = extract_no_meal_data(insulin_data, cgm_data)

    meal_features = extract_features(meal_data_matrix)
    no_meal_features = extract_features(no_meal_data_matrix)
    consolidate_features = meal_features.append(no_meal_features)

    return consolidate_features, len(meal_features), len(no_meal_features)

# Fill NaN values with the column mean or median before training
def fill_na(data):
    return data.fillna(data.mean())

# Main function orchestrating the process
def main():
    patient1_insulin_file = 'InsulinData.csv'
    patient1_cgm_file = 'CGMData.csv'

    patient2_cgm_file = 'CGM_patient2.csv'
    patient2_insulin_file = 'Insulin_patient2.csv'

    # Train the model and save the results
    patient_1_features, patient_1_num_meal_data, patient_1_num_no_meal_data = consolidate_features(patient1_insulin_file, patient1_cgm_file)
    patient_2_features, patient_2_num_meal_data, patient_2_num_no_meal_data = consolidate_features(patient2_insulin_file, patient2_cgm_file)
    patient_1_features = fill_na(patient_1_features)
    patient_2_features = fill_na(patient_2_features)
    features_to_train = pd.concat([patient_1_features, patient_2_features], ignore_index=True)

    # Train the model and save the results
    model, scaler, sampler = train_model(
        features_to_train, 
        patient_1_num_meal_data + patient_2_num_meal_data, 
        patient_1_num_no_meal_data + patient_2_num_no_meal_data
    )


    # Save the trained model and scaler to a pickle file
    with open('model_scaler_sampler.pkl', 'wb') as f:
        pickle.dump((model, scaler, sampler), f)

if __name__ == "__main__":
    main()
