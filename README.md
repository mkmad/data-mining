# Data Mining Projects

This repository contains comprehensive data mining projects focusing on diabetes management through Continuous Glucose Monitoring (CGM) and insulin data analysis. The projects demonstrate various machine learning techniques for meal detection, clustering, and predictive modeling.

## 📁 Project Structure

```
data-mining/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── run_examples.py                     # Interactive runner script
├── 
├── 📊 datasets/                        # Data files
│   ├── README.md                      # Dataset descriptions
│   ├── CGMData.csv                    # Continuous glucose monitoring data
│   ├── CGMData_1.csv                  # Additional CGM dataset
│   ├── CGM_patient2.csv               # Patient 2 CGM data
│   ├── InsulinData.csv                # Insulin administration data
│   ├── InsulinData_2.csv              # Additional insulin dataset
│   ├── Insulin_patient2.csv           # Patient 2 insulin data
│   ├── Results.csv                    # Clustering results
│   └── Final_Results.csv              # Final project results
│   
├── 🔬 projects/                        # Data mining projects
│   ├── README.md                      # Projects overview
│   ├── meal-detection/                # Meal detection project
│   │   ├── train-model.py             # Model training script
│   │   ├── predict.py                 # Prediction script
│   │   ├── model_scaler_sampler.pkl   # Trained model and preprocessing
│   │   └── README.md                  # Meal detection documentation
│   ├── time-series-analysis/          # Time series analysis project
│   │   ├── time-series-extraction.py  # Time series feature extraction
│   │   └── README.md                  # Time series documentation
│   └── clustering-validation/         # Clustering validation project
│       ├── cluster_validation.py      # Clustering analysis script
│       └── README.md                  # Clustering documentation
│   
└── 📖 documentation/                   # Project documentation
    ├── README.md                      # Documentation overview
    ├── train-model.pdf                # Model training guide
    ├── time-series-extraction.pdf     # Time series analysis guide
    └── ClusterValidationProject.pdf   # Clustering validation guide
```

## 🚀 Quick Start Guide

### Option 1: Interactive Runner (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive menu
python run_examples.py
```

### Option 2: Manual Navigation
```bash
# Install dependencies
pip install -r requirements.txt

# Run meal detection project
cd projects/meal-detection
python train-model.py
python predict.py

# Run time series analysis
cd ../time-series-analysis
python time-series-extraction.py

# Run clustering validation
cd ../clustering-validation
python cluster_validation.py
```

## 🎯 Learning Path

### Beginner Level
1. **Start with**: `documentation/train-model.pdf`
2. **Practice**: `projects/meal-detection/train-model.py`
3. **Learn**: Basic machine learning concepts

### Intermediate Level
1. **Study**: `documentation/time-series-extraction.pdf`
2. **Practice**: `projects/time-series-analysis/time-series-extraction.py`
3. **Explore**: Time series feature extraction

### Advanced Level
1. **Master**: `documentation/ClusterValidationProject.pdf`
2. **Practice**: `projects/clustering-validation/cluster_validation.py`
3. **Complete**: Advanced clustering analysis

## 📊 Project Contents

### 1. Meal Detection Project
- **Objective**: Predict meal vs. no-meal events from glucose data
- **Techniques**: Feature extraction, SMOTE oversampling, ensemble methods
- **Models**: Gradient Boosting, Random Forest, SVM, Decision Trees
- **Applications**: Diabetes management, automated insulin delivery

### 2. Time Series Analysis Project
- **Objective**: Extract meaningful features from glucose time series
- **Techniques**: Statistical metrics, day/night segmentation, glucose variability
- **Applications**: Glucose monitoring, treatment optimization

### 3. Clustering Validation Project
- **Objective**: Validate clustering algorithms for glucose patterns
- **Techniques**: K-means, DBSCAN, purity/entropy metrics
- **Applications**: Pattern recognition, patient stratification

## 🎯 Learning Objectives

This collection covers:

1. **Data Preprocessing**
   - Time series data handling
   - Feature engineering
   - Data cleaning and validation

2. **Machine Learning**
   - Supervised learning (classification)
   - Unsupervised learning (clustering)
   - Model evaluation and validation

3. **Advanced Techniques**
   - SMOTE for imbalanced data
   - Cross-validation strategies
   - Ensemble methods

4. **Domain-Specific Analysis**
   - Medical data analysis
   - Time series feature extraction
   - Statistical validation

## 🔧 Key Dependencies

- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing and statistics
- **imbalanced-learn**: Handling imbalanced datasets

## 📚 Course Context

This repository contains materials from **Data Mining** course, covering:

- **Project 1**: Meal Detection using Machine Learning
- **Project 2**: Time Series Feature Extraction
- **Project 3**: Clustering Validation and Analysis

## 🎯 Expected Outcomes

After completing these projects, you should be able to:
1. Preprocess and analyze medical time series data
2. Apply machine learning to classification problems
3. Handle imbalanced datasets effectively
4. Validate clustering algorithms
5. Extract meaningful features from time series
6. Interpret results in medical context

## 📝 Usage Examples

### Running Meal Detection
```python
# Train the model
cd projects/meal-detection
python train-model.py

# Make predictions
python predict.py
```

### Running Time Series Analysis
```python
# Extract time series features
cd projects/time-series-analysis
python time-series-extraction.py
```

### Running Clustering Validation
```python
# Perform clustering analysis
cd projects/clustering-validation
python cluster_validation.py
```

## 🔬 Dataset Descriptions

### CGM Data (Continuous Glucose Monitoring)
- **Purpose**: Real-time glucose level monitoring
- **Features**: Timestamp, glucose levels (mg/dL)
- **Frequency**: Every 5 minutes
- **Applications**: Diabetes management, treatment optimization

### Insulin Data
- **Purpose**: Insulin administration records
- **Features**: Timestamp, carbohydrate input, insulin doses
- **Applications**: Meal detection, insulin therapy

## 🤝 Contributing

Feel free to explore, modify, and extend these projects for your own learning and research purposes.

## 📄 License

This project is for educational purposes. Please respect the original course materials and datasets.

---

**Note**: This project involves medical data analysis. The techniques demonstrated are for educational purposes and should not be used for clinical decision-making without proper validation.