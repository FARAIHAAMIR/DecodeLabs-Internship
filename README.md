# DecodeLabs-Internship
Enterprise-EDA-Feature-Engineering
# Enterprise-Grade Data Engineering & Advanced EDA Pipeline

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.0-green)
![Pandera](https://img.shields.io/badge/Pandera-Data Validation-orange)

## 📌 Overview
This repository contains an Enterprise-Grade Data Preprocessing and Feature Engineering Pipeline developed as part of the **DecodeLabs Industrial Training (Batch 2026)**[cite: 1]. 

Machine Learning estimators possess zero qualitative reasoning—they are numerical optimization algorithms operating on real-numbered coordinate spaces[cite: 1]. Low-fidelity data leads to sub-optimal optimization[cite: 1]. This pipeline implements mathematical controls to clean raw data, prevent silent data corruption, and ensure runtime schema fidelity[cite: 1].

---

## 🏗️ Architecture (Input-Process-Output Framework)

### 1. Module 1: Input (Securing Fidelity)
* **Statistical Imputation Matrix**: Handles missing data using structural thresholds[cite: 1]:
  * `< 5%`: Drop rows[cite: 1].
  * `5% - 20%`: Global Median Imputation[cite: 1].
  * `> 20%`: K-Nearest Neighbors (KNN) Multi-Dimensional Imputation[cite: 1].
* **Outlier Neutralization**: Neutralizes extreme anomalies using Interquartile Range (IQR) Winsorization (`numpy.clip()`) to preserve row count and sequential integrity[cite: 1].

### 2. Module 2: Process (The Vectorized Compute Engine)
* **Categorical Mapping**: Converts nominal features into orthogonal coordinate space via One-Hot Encoding[cite: 1].
* **Feature Engineering**: Vectorized mathematical transformations generating new predictive ratios (`income_per_age`, `score_age_ratio`, `income_score_prod`)[cite: 1].
* **Collinearity Eradication**: Detects highly correlated pairs (`r > 0.80`) and systematically drops redundant features based on target variable correlation to maintain matrix invertibility[cite: 1].

### 3. Module 3: Output (Contracts & Validation)
* **Pandera Data Contracts**: Implements runtime schema assertions to validate data types and boundary conditions, eliminating training-serving skew[cite: 1].

---

## 🛠️ Tech Stack & Requirements
* **Python 3.x**
* **Pandas**
* **NumPy**
* **Scikit-Learn**
* **Pandera**

---

## 🚀 How to Run

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Enterprise-EDA-Feature-Engineering.git](https://github.com/YOUR_USERNAME/Enterprise-EDA-Feature-Engineering.git)
   cd Enterprise-EDA-Feature-Engineering
   pip install pandas numpy scikit-learn pandera
   python project1.py
