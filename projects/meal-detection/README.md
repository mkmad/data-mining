# Meal Detection Project

This project implements machine learning algorithms to detect meal events from Continuous Glucose Monitoring (CGM) data, a crucial component in automated insulin delivery systems.

## 🎯 Project Objective

Predict whether a given glucose time series segment corresponds to a meal event or not, using features extracted from CGM data patterns.

## 📊 Methodology

### Data Processing Pipeline

#### 1. Data Loading
- **Input**: CGM and insulin CSV files
- **Processing**: Parse timestamps and align data streams
- **Output**: Synchronized glucose and insulin data

#### 2. Meal Data Extraction
```python
def extract_meal_data(insulin_data, cgm_data):
    # Extract 2.5-hour glucose segments around meal times
    # Apply meal timing rules and validation
    # Return standardized meal data matrix
```

**Meal Timing Rules**:
- **Condition A**: Next meal within 2 hours → Use next meal time
- **Condition B**: Next meal at exactly 2 hours → Use 1.5-4 hour window
- **Condition C**: No meal within 2 hours → Use 0.5-2 hour window

#### 3. No-Meal Data Extraction
```python
def extract_no_meal_data(insulin_data, cgm_data):
    # Extract glucose segments between meals
    # Ensure minimum 2-hour gaps from meals
    # Return standardized no-meal data matrix
```

#### 4. Feature Engineering
Extract 24 comprehensive features from each glucose segment:

**Statistical Features**:
- Mean, standard deviation, min, max, median
- Variance, range, skewness, kurtosis
- Percentiles (10th, 25th, 75th, 90th)

**Temporal Features**:
- Rate of change (mean, max, min differences)
- First/second half statistics
- Peak analysis (count, amplitude)

**Rolling Features**:
- 5-point rolling mean and standard deviation

#### 5. Data Balancing
- **Problem**: Imbalanced dataset (more no-meal than meal events)
- **Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Result**: Balanced training dataset

#### 6. Model Training
Train multiple algorithms with cross-validation:

**Models Used**:
- **Gradient Boosting Classifier**: Primary model
- **Random Forest Classifier**: Ensemble method
- **Support Vector Machine**: Linear and RBF kernels
- **Decision Tree**: Baseline comparison

**Validation Strategy**:
- **Stratified K-Fold Cross-Validation**: 5 folds
- **Grid Search**: Hyperparameter optimization
- **Performance Metrics**: Accuracy, precision, recall, F1-score

## 🚀 Usage

### Prerequisites
```bash
pip install scikit-learn pandas numpy scipy imbalanced-learn
```

### Training the Model
```bash
python train-model.py
```

**Input Requirements**:
- `../datasets/CGMData.csv` - Glucose monitoring data
- `../datasets/InsulinData.csv` - Insulin administration data

**Output**:
- `model_scaler_sampler.pkl` - Trained model and preprocessing components

### Making Predictions
```bash
python predict.py
```

**Input Requirements**:
- `model_scaler_sampler.pkl` - Trained model
- `test.csv` - Test data in N×24 matrix format

**Output**:
- `Result.csv` - Binary predictions (0=no-meal, 1=meal)

## 📈 Model Performance

### Expected Performance Metrics
- **Accuracy**: 70-85%
- **Precision**: 65-80%
- **Recall**: 70-85%
- **F1-Score**: 70-85%

### Performance Factors
- **Data Quality**: Sensor accuracy and calibration
- **Feature Engineering**: Relevance of extracted features
- **Class Imbalance**: Effectiveness of SMOTE
- **Temporal Patterns**: Consistency of meal timing

## 🔬 Technical Details

### Data Requirements
- **CGM Data**: 5-minute interval glucose readings
- **Insulin Data**: Meal events with carbohydrate input
- **Minimum Duration**: 2.5 hours of continuous data
- **Data Quality**: <20% missing values

### Feature Extraction Process
1. **Normalization**: Standardize glucose values
2. **Segmentation**: Extract 24-point windows
3. **Feature Computation**: Calculate 24 statistical features
4. **Validation**: Ensure feature quality and relevance

### Model Selection Criteria
- **Cross-Validation Score**: Primary selection metric
- **Computational Efficiency**: Training and prediction speed
- **Interpretability**: Feature importance analysis
- **Robustness**: Performance on validation sets

## 📊 Results Interpretation

### Prediction Output
- **0**: No meal detected
- **1**: Meal detected

### Confidence Assessment
- **High Confidence**: Clear glucose pattern changes
- **Low Confidence**: Ambiguous or noisy patterns
- **Uncertainty**: Missing or incomplete data

### Clinical Relevance
- **True Positives**: Correctly identified meals
- **False Positives**: Incorrect meal detection
- **True Negatives**: Correctly identified no-meal periods
- **False Negatives**: Missed meal events

## 🔧 Customization

### Feature Engineering
Modify `extract_features()` function to:
- Add domain-specific features
- Change statistical measures
- Adjust temporal windows
- Include external factors

### Model Configuration
Adjust hyperparameters in `train_model()`:
- **Gradient Boosting**: n_estimators, learning_rate, max_depth
- **Random Forest**: n_estimators, max_features, max_depth
- **SVM**: C, gamma, kernel
- **Cross-Validation**: n_splits, scoring metrics

### Data Preprocessing
Customize preprocessing steps:
- **Missing Data**: Interpolation vs. removal
- **Outlier Detection**: Statistical vs. domain-based
- **Normalization**: StandardScaler vs. MinMaxScaler
- **Balancing**: SMOTE vs. other techniques

## 📝 Best Practices

### Data Quality
- Validate sensor calibration
- Check for systematic errors
- Ensure temporal alignment
- Handle missing data appropriately

### Model Development
- Use appropriate validation strategies
- Avoid data leakage
- Document preprocessing steps
- Monitor for overfitting

### Clinical Application
- Validate with domain experts
- Consider individual variations
- Account for external factors
- Maintain ethical standards

## 🔗 Dependencies

### Required Libraries
```python
import pandas as pd
import numpy as np
from scipy.fft import fft
import scipy.stats
from scipy.signal import find_peaks
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
```

### Data Dependencies
- CGM and insulin datasets in `../datasets/`
- Proper data format and structure
- Sufficient data quality for analysis

## 🎯 Future Improvements

### Potential Enhancements
1. **Deep Learning**: LSTM/CNN for temporal patterns
2. **Ensemble Methods**: Combine multiple models
3. **Real-time Processing**: Online learning capabilities
4. **Personalization**: Patient-specific models
5. **External Factors**: Activity, stress, medication

### Research Directions
- **Feature Selection**: Automated feature importance
- **Model Interpretability**: Explainable AI techniques
- **Clinical Validation**: Real-world performance assessment
- **Integration**: Automated insulin delivery systems

---

**Note**: This model is for educational and research purposes. Clinical applications require extensive validation and regulatory approval.
