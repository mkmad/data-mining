import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

insulin_data = pd.read_csv('InsulinData.csv', low_memory=False)
cgm_data = pd.read_csv('CGMData.csv', low_memory=False)

# Extract and bin carbohydrate input
carb_input_data = insulin_data['BWZ Carb Input (grams)'].dropna().astype(float)
carb_input_data = carb_input_data[carb_input_data > 0]
carb_min = carb_input_data.min()
carb_max = carb_input_data.max()
bin_size = 20
num_bins = (carb_max - carb_min) // bin_size + 1
carb_bins = pd.cut(carb_input_data, bins=int(num_bins), labels=False, right=False) * bin_size + carb_min

ground_truth_labels = np.random.randint(0, int(num_bins), len(carb_input_data))  

# Prepare CGM data and extract meal data segments
cgm_data['Datetime'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
cgm_data = cgm_data.sort_values('Datetime')

# Randomly sample meal times
meal_times = cgm_data['Datetime'].sample(10).sort_values()

# Extract glucose data segments based on meal times
meal_data_segments = []
for tm in meal_times:
    tm_plus_2hrs = tm + pd.Timedelta(hours=2)
    tm_minus_30min = tm - pd.Timedelta(minutes=30)
    meal_within_window = meal_times[(meal_times > tm) & (meal_times < tm_plus_2hrs)]
    if not meal_within_window.empty:
        tp = meal_within_window.min()
        segment = cgm_data[(cgm_data['Datetime'] >= tp) & (cgm_data['Datetime'] <= tp + pd.Timedelta(hours=2))]
    else:
        segment = cgm_data[(cgm_data['Datetime'] >= tm_minus_30min) & (cgm_data['Datetime'] <= tm_plus_2hrs)]
    meal_data_segments.append(segment['Sensor Glucose (mg/dL)'].dropna().values)

# Extract features and standardize
def extract_features(segments):
    features = []
    for segment in segments:
        if segment.size > 0:
            features.append([np.mean(segment), np.std(segment), np.min(segment), np.max(segment)])
        else:
            features.append([np.nan, np.nan, np.nan, np.nan])
    return np.array(features)

features = extract_features(meal_data_segments)
features = np.nan_to_num(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Clustering
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
dbscan = DBSCAN(eps=1.5, min_samples=2)
kmeans_labels = kmeans.fit_predict(features_scaled)
dbscan_labels = dbscan.fit_predict(features_scaled)

# Compute SSE
kmeans_sse = kmeans.inertia_
dbscan_centroids = [features_scaled[dbscan_labels == label].mean(axis=0) for label in np.unique(dbscan_labels) if label != -1]
dbscan_sse = sum(np.sum((features_scaled[dbscan_labels == label] - centroid) ** 2) for label, centroid in zip(np.unique(dbscan_labels), dbscan_centroids))

# Calculate purity and entropy
def calculate_purity_entropy(cluster_labels, true_labels, num_clusters, num_bins):
    confusion_matrix = np.zeros((num_clusters, num_bins))
    for i in range(len(cluster_labels)):
        confusion_matrix[cluster_labels[i], true_labels[i]] += 1
    purity = np.sum(np.max(confusion_matrix, axis=1)) / np.sum(confusion_matrix)
    entropy_vals = entropy(confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True), base=2, axis=1)
    weighted_entropy = np.sum(entropy_vals * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix))
    return purity, weighted_entropy

purity_kmeans, entropy_kmeans = calculate_purity_entropy(kmeans_labels, ground_truth_labels[:len(kmeans_labels)], n_clusters, int(num_bins))
purity_dbscan, entropy_dbscan = calculate_purity_entropy(dbscan_labels, ground_truth_labels[:len(dbscan_labels)], len(np.unique(dbscan_labels)), int(num_bins))

# Save results
results_df = pd.DataFrame({
    "SSE for KMeans": [kmeans_sse],
    "SSE for DBSCAN": [dbscan_sse],
    "Entropy for KMeans": [entropy_kmeans],
    "Entropy for DBSCAN": [entropy_dbscan],
    "Purity for KMeans": [purity_kmeans],
    "Purity for DBSCAN": [purity_dbscan]
})
results_df.to_csv('Result.csv', header=False, index=False)
