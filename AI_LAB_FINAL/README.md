# AI Lab Final Exam - Machine Learning & Deep Learning

**Student:** Abdul Rehman Saeed  
**Registration:** FA22-BCS-055  
**Course:** Artificial Intelligence Lab  

---

## 📋 Overview

This repository contains solutions to the AI Lab Final Exam, covering: 
- **Question 1:** Machine Learning (Titanic Dataset - Random Forest Classifier)
- **Question 2:** Deep Learning (CIFAR-10 CNN Image Classification)

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `AI LAB FINAL.pdf` | Exam questions and requirements |
| `AI_Lab_Final.ipynb` | Complete solutions notebook (both questions) |
| `README.md` | This file |

---

## 🚀 Question 1: Titanic Survival Prediction

### Objective
Build a Machine Learning model to predict passenger survival on the Titanic using the Random Forest Classifier.

### Dataset
- **Source:** [Titanic Dataset (DataScienceDojo)](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Records:** 891 passengers
- **Features:** 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, etc.)

### Implementation Steps

1. **Data Loading & Exploration**
   - Loaded dataset using pandas
   - Inspected data structure, statistics, and missing values

2. **Data Preprocessing**
   - **Missing Values:**
     - Age:  Filled with median (28. 0 years)
     - Embarked: Filled with mode ('S')
     - Cabin: Dropped (too many missing values)
   - **Feature Engineering:**
     - Dropped unnecessary columns:  PassengerId, Name, Ticket, Cabin
     - Encoded Sex:  male=1, female=0
     - Encoded Embarked using LabelEncoder
   - **Normalization:**
     - Standardized features using StandardScaler

3. **Model Training**
   - **Algorithm:** Random Forest Classifier
   - **Parameters:** 100 trees, max_depth=10
   - **Split:** 80% training, 20% testing (stratified)

4. **Evaluation Metrics**

| Metric | Score |
|--------|-------|
| **Accuracy** | ~83% |
| **Precision** | ~80% |
| **Recall** | ~75% |
| **F1-Score** | ~77% |

5. **Feature Importance**
   - Most important features: Sex, Fare, Age, Pclass

### Key Results
✅ Successfully handled missing data  
✅ Achieved 83% accuracy on test set  
✅ Visualized confusion matrix and feature importance  

---

## 🧠 Question 2: CIFAR-10 Image Classification

### Objective
Build a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

### Dataset
- **Source:** CIFAR-10 (built into TensorFlow/Keras)
- **Training:** 50,000 images (32x32 RGB)
- **Testing:** 10,000 images
- **Classes:** 10 (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)

### CNN Architecture

```
Input:  32x32x3 RGB images

Block 1:
  - Conv2D(32 filters, 3x3) + ReLU + BatchNorm
  - Conv2D(32 filters, 3x3) + ReLU + BatchNorm
  - MaxPooling(2x2)
  - Dropout(0.25)

Block 2:
  - Conv2D(64 filters, 3x3) + ReLU + BatchNorm
  - Conv2D(64 filters, 3x3) + ReLU + BatchNorm
  - MaxPooling(2x2)
  - Dropout(0.25)

Block 3:
  - Conv2D(128 filters, 3x3) + ReLU + BatchNorm
  - Conv2D(128 filters, 3x3) + ReLU + BatchNorm
  - MaxPooling(2x2)
  - Dropout(0.25)

Fully Connected:
  - Flatten
  - Dense(128) + ReLU + BatchNorm
  - Dropout(0.5)
  - Dense(10) + Softmax

Total Parameters: ~500,000 trainable parameters
```

### Training Configuration
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Epochs:** 50 (with Early Stopping)
- **Batch Size:** 128
- **Callbacks:**
  - Early Stopping (patience=10)
  - ReduceLROnPlateau (factor=0.5, patience=5)

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~75% |
| **Training Loss** | ~0.65 |
| **Validation Loss** | ~0.75 |

### Visualizations
- ✅ Sample images from dataset
- ✅ Class distribution bar chart
- ✅ Training/Validation Loss vs Epochs
- ✅ Training/Validation Accuracy vs Epochs
- ✅ Sample predictions with confidence scores

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML algorithms & preprocessing |
| **TensorFlow/Keras** | Deep learning framework |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical plots |

---

## 📊 Key Achievements

### Question 1 (Titanic)
✅ **83% accuracy** with Random Forest  
✅ Proper handling of missing data  
✅ Feature importance analysis  
✅ Confusion matrix visualization  

### Question 2 (CIFAR-10)
✅ **~75% accuracy** on image classification  
✅ Custom CNN with 3 convolutional blocks  
✅ Early stopping & learning rate scheduling  
✅ Model saved in .h5 and .keras formats  

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Run Jupyter Notebook
```bash
jupyter notebook AI_Lab_Final.ipynb
```

### Or Use Google Colab
1. Upload `AI_Lab_Final.ipynb` to Google Colab
2. Enable GPU:  `Runtime` → `Change runtime type` → `GPU`
3. Run all cells

---

## 📝 Summary

This project demonstrates:
- **Data preprocessing** techniques (handling missing values, encoding, normalization)
- **Machine Learning** model training (Random Forest)
- **Deep Learning** CNN architecture design
- **Model evaluation** with multiple metrics
- **Visualization** of results and model performance

### Performance Comparison

| Task | Algorithm | Accuracy |
|------|-----------|----------|
| Titanic Survival | Random Forest | **83%** |
| CIFAR-10 Classification | CNN | **~75%** |

---

## 👤 Author
**Abdul Rehman Saeed**  
Registration: FA22-BCS-055  
University: [Your University Name]  
Course: Artificial Intelligence Lab  

---

## 📄 License
This project is for educational purposes (AI Lab Final Exam).

---

## 🙏 Acknowledgments
- Titanic Dataset: [Kaggle/DataScienceDojo](https://www.kaggle.com/c/titanic)
- CIFAR-10 Dataset: [Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html)
- TensorFlow/Keras Documentation
- Scikit-learn Documentation

---

**Status:** ✅ Completed & Verified  
**Date:** December 2025  
