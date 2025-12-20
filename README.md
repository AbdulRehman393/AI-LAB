# 🤖 AI/ML/DL Laboratory - Complete Learning Path

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)]()

> A comprehensive collection of **Artificial Intelligence**, **Machine Learning**, and **Deep Learning** lab experiments, implementations, and projects — from Python basics to advanced neural networks. 

---

## 📋 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Topics Covered](#topics-covered)
- [Key Projects](#key-projects)
- [Technologies & Tools](#technologies--tools)
- [Getting Started](#getting-started)
- [Author](#author)

---

## 🎯 Overview

This repository documents my journey through **Artificial Intelligence Lab** coursework at **COMSATS University Islamabad**, featuring: 

✅ **15+ Lab Experiments**  
✅ **Classical AI Search Algorithms**  
✅ **Machine Learning Models** (Scikit-learn)  
✅ **Deep Learning Networks** (TensorFlow/Keras)  
✅ **Real-World Applications** (Industrial Threat Detection, Image Classification)  
✅ **95%+ Jupyter Notebooks** for interactive learning

---

## 📂 Repository Structure

```
AI-LAB/
│
├── 📘 AI Important Concepts Lab Assign 1/    # Fundamental AI concepts
├── 🔍 AI-Lab-2-main/                         # Search algorithms (BFS, DFS, A*)
├── 🧪 AI LAB 3/                              # Mid-level ML experiments
├── 🧪 AI LAB 4/                              # Advanced ML techniques
├── 📝 AI MID LAB/                            # Midterm practical work
├── 🎓 AI_LAB_FINAL/                          # Final exam projects
│   ├── Titanic Survival Prediction (83% accuracy)
│   └── CIFAR-10 CNN Classification (~75% accuracy)
│
├── 🎮 Tic_Tac_Toe/                           # Minimax & Alpha-Beta Pruning
├── 🔐 Lab_02_Threat_Detection/               # ICS Anomaly Detection (SWaT dataset)
├── 🧠 BackPropagation/                       # Neural network fundamentals
├── 🌐 RNN RMDB dataset/                      # Recurrent networks for NLP
├── 🤖 FuzzyReasoning/                        # Fuzzy logic implementation
├── 📊 Knn_classifiaction/                    # K-Nearest Neighbors
│
├── Lab1ipynb.ipynb                           # Python basics
├── AlphabetaPruning.ipynb                    # Game tree optimization
└── README.md                                 # This file
```

---

## 🧠 Topics Covered

### 1️⃣ **Python Fundamentals**
- Variables, Data Types & Operators
- Control Structures (Loops, Conditionals)
- Functions & File Handling
- Data Structures (Lists, Dictionaries, Sets)

### 2️⃣ **Classical AI Search Algorithms**
- **Uninformed Search**: BFS, DFS, Uniform Cost Search
- **Informed Search**: A* Search with Heuristics
- **Game Theory**: Minimax, Alpha-Beta Pruning
- **Applications**: Pathfinding, Tic-Tac-Toe AI

### 3️⃣ **Machine Learning**
| Algorithm | Use Case | Dataset |
|-----------|----------|---------|
| **Random Forest** | Classification | Titanic Survival (83% accuracy) |
| **K-Nearest Neighbors** | Classification | Custom datasets |
| **Linear Regression** | Prediction | Various numerical datasets |
| **Decision Trees** | Classification | Multi-class problems |

**Tools**:  Scikit-learn, Pandas, NumPy, Matplotlib

### 4️⃣ **Deep Learning**
| Architecture | Application | Performance |
|--------------|-------------|-------------|
| **CNN** | CIFAR-10 Image Classification | ~75% accuracy |
| **RNN/LSTM** | Sequence & Text Processing | IMDB sentiment analysis |
| **Autoencoders** | Anomaly Detection | SWaT dataset |
| **ANN** | General Classification | Various tasks |

**Frameworks**: TensorFlow, Keras

### 5️⃣ **Advanced Topics**
- 🔐 **Threat Detection**:  LSTM Autoencoder + Isolation Forest for Industrial Control Systems
- 🧮 **Fuzzy Logic**: Fuzzy reasoning systems
- 🎲 **Ensemble Methods**: Hybrid model combining ML + DL

---

## 🏆 Key Projects

### 🚢 **1. Titanic Survival Prediction**
**Folder**: `AI_LAB_FINAL/`

- **Algorithm**: Random Forest Classifier
- **Accuracy**: **83%**
- **Highlights**:
  - Handled missing data (Age, Embarked, Cabin)
  - Feature engineering (Sex encoding, dropping irrelevant columns)
  - StandardScaler normalization
  - Confusion matrix & feature importance visualization

[📄 Detailed Documentation](./AI_LAB_FINAL/README. md)

---

### 🖼️ **2. CIFAR-10 Image Classification**
**Folder**: `AI_LAB_FINAL/`

- **Architecture**: Custom 3-block CNN
  - 3 Convolutional Blocks (32→64→128 filters)
  - Batch Normalization + Dropout
  - MaxPooling & Dense layers
- **Accuracy**: **~75%**
- **Highlights**:
  - 50,000 training images (32×32 RGB)
  - Early stopping & learning rate scheduling
  - Model saved in `.h5` and `.keras` formats

[📄 Detailed Documentation](./AI_LAB_FINAL/README.md)

---

### 🔐 **3. Industrial Control System Threat Detection**
**Folder**: `Lab_02_Threat_Detection/`

- **Dataset**: SWaT (Secure Water Treatment) — 14,997 samples, 78 features
- **Models**:
  1. **Isolation Forest** (ML-based anomaly detection)
  2. **LSTM Autoencoder** (DL-based sequence modeling)
  3. **Hybrid Ensemble** (Combined approach)
- **Use Case**: Cyber-physical system security

**Links**:
- [Dataset (Google Drive)](https://drive.google.com/drive/folders/1PVA1ccYj5S6LTm8bpDG9b7JroI3Ues7w? usp=sharing)
- [Report (Overleaf)](https://www.overleaf.com/read/gdkvjvmztqyn#e243c1)

[📄 Detailed Documentation](./Lab_02_Threat_Detection/README.md)

---

### 🎮 **4. Tic-Tac-Toe AI**
**Folder**: `Tic_Tac_Toe/`

- Minimax algorithm implementation
- Alpha-Beta pruning optimization
- Unbeatable AI opponent

---

## ⚙️ Technologies & Tools

| Category | Technologies |
|: --------:|: -------------|
| **Language** | Python 3.x |
| **Data Science** | NumPy, Pandas, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Deep Learning** | TensorFlow, Keras |
| **Development** | Jupyter Notebook, Google Colab |
| **Version Control** | Git, GitHub |

---

## 🚀 Getting Started

### Prerequisites
```bash
# Install required libraries
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```

### Usage Options

#### **Option 1: Google Colab (Recommended)**
1. Click any `.ipynb` file in the repository
2. Click "Open in Colab" button
3. Enable GPU:  `Runtime` → `Change runtime type` → `GPU`
4. Run all cells

#### **Option 2: Local Jupyter Notebook**
```bash
# Clone repository
git clone https://github.com/AbdulRehman393/AI-LAB.git
cd AI-LAB

# Install Jupyter
pip install jupyter

# Launch notebook
jupyter notebook
```

#### **Option 3: Browse Individual Labs**
Navigate to specific folders:
- `AI_LAB_FINAL/` → Final exam projects
- `Lab_02_Threat_Detection/` → Threat detection project
- `Tic_Tac_Toe/` → Game AI implementation

---

## 📈 Learning Outcomes

By exploring this repository, you will understand: 

✅ **Python Programming**:  From basics to advanced data manipulation  
✅ **AI Search Techniques**:  Pathfinding and optimization algorithms  
✅ **ML Pipeline**:  Data preprocessing → Model training → Evaluation  
✅ **Deep Learning**: CNN/RNN architectures for image & sequence data  
✅ **Real-World Applications**: Cybersecurity, healthcare, computer vision  
✅ **Best Practices**: Code organization, documentation, reproducibility

---

## 📊 Performance Summary

| Project | Algorithm | Dataset | Accuracy | Type |
|---------|-----------|---------|----------|------|
| Titanic Survival | Random Forest | Kaggle Titanic | 83% | ML |
| CIFAR-10 Classification | CNN | CIFAR-10 | ~75% | DL |
| Threat Detection | Hybrid (IF+LSTM) | SWaT | N/A | ML+DL |
| Tic-Tac-Toe | Minimax + α-β | Game States | Unbeatable | AI |

---

## 🧑‍💻 Author

**Abdul Rehman Saeed**  
📧 Registration: FA22-BCS-055  
🎓 COMSATS University Islamabad, Abbottabad Campus  
🌐 [GitHub Profile](https://github.com/AbdulRehman393)  
💼 AI Enthusiast | Machine Learning Developer

---

## 📄 License

This repository is for **educational purposes** as part of university coursework.   
Feel free to explore, learn, and adapt the code for your own educational projects.

---

## 🙏 Acknowledgments

- **Datasets**:  Kaggle, UCI ML Repository, iTrust Labs Singapore
- **Frameworks**: TensorFlow, Scikit-learn communities
- **Inspiration**:  COMSATS AI Lab instructors and peers

---

## 🔗 Quick Links

- [Final Exam Projects](./AI_LAB_FINAL/)
- [Threat Detection Project](./Lab_02_Threat_Detection/)
- [Search Algorithms](./AI-Lab-2-main/)
- [Game AI](./Tic_Tac_Toe/)

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue? style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
</p>

<p align="center">
  ⭐ If you find this repository helpful, please consider starring it!
</p>
