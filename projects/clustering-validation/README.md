# Clustering Validation Project

This project implements and validates clustering algorithms for glucose pattern recognition, comparing K-means and DBSCAN algorithms using purity and entropy metrics.

## 🎯 Project Objective

Validate clustering algorithms for identifying patterns in glucose data segments, with particular focus on meal-related glucose patterns and carbohydrate intake clustering.

## 📊 Methodology

### Data Processing Pipeline

#### 1. Data Loading and Preprocessing
```python
# Load insulin and CGM data
insulin_data = pd.read_csv('InsulinData.csv', low_memory=False)
cgm_data = pd.read_csv('CGMData.csv', low_memory=False)
```

**Data Sources**:
- **Insulin Data**: Carbohydrate intake and administration records
- **CGM Data**: Continuous glucose monitoring readings

#### 2. Carbohydrate Input Binning
```python
# Extract and bin carbohydrate input
carb_input_data = insulin_data['BWZ Carb Input (grams)'].dropna().astype(float)
carb_input_data = carb_input_data[carb_input_data > 0]
bin_size = 20
num_bins = (carb_max - carb_min) // bin_size + 1
carb_bins = pd.cut(carb_input_data, bins=int(num_bins), labels=False, right=False) * bin_size + carb_min
```

**Binning Strategy**:
- **Bin Size**: 20 grams per bin
- **Range**: From minimum to maximum carbohydrate intake
- **Purpose**: Create ground truth labels for validation

#### 3. Meal Data Segment Extraction
```python
# Extract glucose data segments based on meal times
meal_data_segments = []
for tm in meal_times:
    # Define time windows around meal events
    # Extract glucose data within windows
    # Store standardized segments
```

**Extraction Criteria**:
- **Time Window**: 2.5 hours around meal events
- **Data Quality**: Minimum segment length requirements
- **Temporal Alignment**: Synchronized with meal timing

#### 4. Feature Extraction
```python
def extract_features(segments):
    features = []
    for segment in segments:
        if segment.size > 0:
            features.append([np.mean(segment), np.std(segment), np.min(segment), np.max(segment)])
        else:
            features.append([np.nan, np.nan, np.nan, np.nan])
    return np.array(features)
```

**Extracted Features**:
- **Mean**: Average glucose level in segment
- **Standard Deviation**: Glucose variability
- **Minimum**: Lowest glucose reading
- **Maximum**: Highest glucose reading

#### 5. Data Standardization
```python
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```

**Standardization Process**:
- **Z-score Normalization**: (x - μ) / σ
- **Purpose**: Ensure features are on same scale
- **Impact**: Improves clustering algorithm performance

#### 6. Clustering Algorithms

**K-means Clustering**:
```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(features_scaled)
```

**DBSCAN Clustering**:
```python
dbscan = DBSCAN(eps=1.5, min_samples=2)
dbscan_labels = dbscan.fit_predict(features_scaled)
```

**Algorithm Parameters**:
- **K-means**: 7 clusters, random initialization
- **DBSCAN**: eps=1.5, min_samples=2

#### 7. Validation Metrics

**Sum of Squared Errors (SSE)**:
```python
kmeans_sse = kmeans.inertia_
dbscan_sse = sum(np.sum((features_scaled[dbscan_labels == label] - centroid) ** 2) 
                for label, centroid in zip(np.unique(dbscan_labels), dbscan_centroids))
```

**Purity and Entropy**:
```python
def calculate_purity_entropy(cluster_labels, true_labels, num_clusters, num_bins):
    confusion_matrix = np.zeros((num_clusters, num_bins))
    for i in range(len(cluster_labels)):
        confusion_matrix[cluster_labels[i], true_labels[i]] += 1
    purity = np.sum(np.max(confusion_matrix, axis=1)) / np.sum(confusion_matrix)
    entropy_vals = entropy(confusion_matrix / np.sum(confusion_matrix, axis=1, keepdims=True), base=2, axis=1)
    weighted_entropy = np.sum(entropy_vals * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix))
    return purity, weighted_entropy
```

## 🚀 Usage

### Prerequisites
```bash
pip install scikit-learn pandas numpy scipy
```

### Running the Validation
```bash
python cluster_validation.py
```

**Input Requirements**:
- `../datasets/InsulinData.csv` - Insulin administration data
- `../datasets/CGMData.csv` - Glucose monitoring data

**Output**:
- `Result.csv` - Clustering validation metrics

## 📈 Validation Metrics

### 1. Sum of Squared Errors (SSE)
**Purpose**: Measure clustering compactness
**Interpretation**: Lower values indicate tighter clusters
**Formula**: Σ(x - centroid)² for all points in each cluster

### 2. Purity
**Purpose**: Measure clustering accuracy against ground truth
**Range**: 0 to 1 (higher is better)
**Formula**: Σ(max(confusion_matrix_row)) / total_points

### 3. Entropy
**Purpose**: Measure clustering uncertainty
**Range**: 0 to log₂(num_classes) (lower is better)
**Formula**: -Σ(p * log₂(p)) for each cluster

## 🔬 Technical Details

### Data Requirements
- **Temporal Resolution**: 5-minute glucose intervals
- **Meal Events**: Carbohydrate intake records
- **Data Quality**: Complete glucose segments
- **Alignment**: Synchronized timestamps

### Feature Engineering
**Statistical Features**:
- **Central Tendency**: Mean glucose levels
- **Variability**: Standard deviation
- **Range**: Min/max glucose values
- **Distribution**: Shape characteristics

**Temporal Features**:
- **Meal Timing**: Time relative to meal events
- **Duration**: Segment length
- **Pattern**: Glucose response patterns

### Clustering Parameters

**K-means Configuration**:
- **Number of Clusters**: 7 (based on carbohydrate bins)
- **Initialization**: Random
- **Convergence**: Maximum iterations or tolerance
- **Random State**: 42 (for reproducibility)

**DBSCAN Configuration**:
- **Epsilon**: 1.5 (neighborhood radius)
- **Min Samples**: 2 (minimum cluster size)
- **Metric**: Euclidean distance
- **Algorithm**: Auto (best algorithm selection)

## 📊 Expected Results

### Performance Comparison
- **K-means**: Typically higher purity, lower entropy
- **DBSCAN**: May identify noise points, variable cluster count
- **SSE**: K-means usually has lower SSE (by design)

### Validation Outcomes
- **Purity**: 0.6-0.9 range expected
- **Entropy**: 0.5-1.5 range expected
- **SSE**: Varies based on data distribution

### Interpretation Guidelines
- **High Purity**: Clusters align well with carbohydrate intake
- **Low Entropy**: Clusters are homogeneous
- **Low SSE**: Clusters are compact and well-defined

## 🔧 Customization

### Feature Engineering
Modify `extract_features()` to include:
- **Additional Statistics**: Skewness, kurtosis, percentiles
- **Temporal Features**: Rate of change, peak analysis
- **Domain Features**: Clinical glucose metrics
- **Derived Features**: Ratios, differences, transformations

### Clustering Parameters
Adjust algorithm settings:
- **K-means**: Number of clusters, initialization method
- **DBSCAN**: Epsilon, min_samples, distance metric
- **Other Algorithms**: Hierarchical, spectral clustering

### Validation Metrics
Customize evaluation approach:
- **Additional Metrics**: Adjusted Rand Index, Silhouette Score
- **Ground Truth**: Different labeling strategies
- **Cross-Validation**: Multiple validation runs
- **Statistical Testing**: Significance testing

## 📝 Best Practices

### Data Preprocessing
- **Quality Control**: Remove incomplete segments
- **Normalization**: Standardize features appropriately
- **Outlier Handling**: Identify and handle outliers
- **Missing Data**: Impute or remove missing values

### Clustering Validation
- **Multiple Metrics**: Use several validation measures
- **Ground Truth**: Ensure meaningful ground truth labels
- **Parameter Tuning**: Optimize algorithm parameters
- **Reproducibility**: Set random seeds for consistency

### Result Interpretation
- **Clinical Relevance**: Consider medical implications
- **Statistical Significance**: Test for meaningful differences
- **Practical Utility**: Assess real-world applicability
- **Limitations**: Acknowledge method constraints

## 🔗 Dependencies

### Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
```

### Data Dependencies
- Insulin and CGM datasets in `../datasets/`
- Proper data format and quality
- Sufficient data for meaningful clustering

## 🎯 Applications

### Clinical Applications
- **Patient Stratification**: Group patients by glucose patterns
- **Treatment Personalization**: Tailor therapy to pattern types
- **Risk Assessment**: Identify high-risk glucose patterns
- **Outcome Prediction**: Predict treatment response

### Research Applications
- **Pattern Discovery**: Identify novel glucose patterns
- **Population Studies**: Analyze pattern distributions
- **Technology Assessment**: Evaluate monitoring systems
- **Quality Improvement**: Monitor diabetes care quality

## 📚 Additional Resources

### Documentation
- Check `../documentation/ClusterValidationProject.pdf` for detailed methodology
- Review clustering algorithm literature
- Consult diabetes pattern analysis guidelines

### Related Projects
- **Meal Detection**: `../meal-detection/` for supervised learning
- **Time Series Analysis**: `../time-series-analysis/` for feature extraction

---

**Note**: This validation is for educational and research purposes. Clinical applications require extensive validation and should be interpreted by healthcare professionals.
