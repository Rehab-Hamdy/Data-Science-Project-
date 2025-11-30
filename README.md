# 📘 Superstore Sales Prediction – Full Machine Learning Project

### Using Random Forest, Decision Tree, SVR, and Neural Network (Keras)

---

# 🚀 1. Project Overview

This project builds a complete **end-to-end regression system** to predict **Sales** using the Superstore dataset.
It follows a full Data Science lifecycle:

* Data Loading
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Preprocessing (encoding, outlier clipping, scaling)
* Training 4 ML models
* Neural Network with GPU acceleration
* Hyperparameter Tuning
* Model Saving (for deployment)
* Deployment-ready pipeline

Every step is implemented exactly as shown in the notebook PDF .

---

# 📂 2. Dataset Description

The dataset contains **9,994 rows × 21 columns**.

### Key Columns

* **Categorical:** Ship Mode, Segment, Region, City, State, Category
* **Numerical:** Sales, Quantity, Discount, Profit
* **Date:** Order Date, Ship Date

### No Missing Values

---

# 📊 3. Exploratory Data Analysis (EDA)


---

# 🛠 4. Feature Engineering & Preprocessing

### ✔ 4.1 Dropping Unnecessary Columns

removed:

```
Row ID, Order ID, Customer ID, Customer Name, Product ID, Country
```

### ✔ 4.2 Extracting Date Features

Converted dates → extracted:

* OrderYear
* OrderMonth
* ShippingDelay


### ✔ 4.3 Categorical Encoding

Label-encoded **8 columns**:

```
Ship Mode, Segment, Region, Category, Sub-Category,
Product Name, City, State
```

Encoders saved as:
`label_encoders.joblib`

### ✔ 4.4 Outlier Clipping

Applied **1–99% clipping** to:

* Profit → Profit_clipped
* Sales → Sales_clipped
* Quantity → Quantity_clipped


### ✔ 4.5 Splitting & Scaling

* 80% train / 20% test
* Scaling:

  * **SVR** → StandardScaler
  * **Neural Network** → Two separate scalers

Saved:

```
standard_scaler_sklearn.joblib
standard_scaler_X_keras.joblib
standard_scaler_y_keras.joblib
```

---

# 🤖 5. Machine Learning Models

Trained and evaluated 4 models:

---

## 5.1 Support Vector Regressor (SVR)


* Kernel: RBF
* Scaled inputs

### Results

```
MAE  = 96.98
MSE  = 62853.12
R²   = 0.6508
```

---

## 5.2 Decision Tree Regressor


* max_depth = 10
* min_samples_leaf = 5

### Results

```
MAE  = 72.54
MSE  = 38164.71
R²   = 0.7879
```

---

## 5.3 Random Forest Regressor


* n_estimators = 500
* max_depth = 15

### Results

```
MAE  = 64.06
MSE  = 31078.00
R²   = 0.8273     ← ⭐ Best Model
```
---

## 5.4 Neural Network (Keras)

Architecture:

```
Dense(64, tanh)
Dense(128, relu)
Dropout(0.3)
Dense(64, relu)
Dropout(0.1)
Dense(32, relu)
Dense(1)
```

### Enhancements

* EarlyStopping (patience=10)
* Adam optimizer
* Dropout regularization
* GPU Acceleration

### Training Curves


* Training vs Validation Loss
* Training vs Validation MAE

### Neural Network Results

```
MAE  = 76.53
MSE  = 32645.24
R²   = 0.8186
```

---

# ⚡ 6. GPU Acceleration (Colab)

GPU details printed on page 1:

```
GPU: Tesla T4
CUDA Version: 12.4
Driver Version: 550.54.15
```



Verified using:

```bash
!nvidia-smi
```

Neural network training was significantly faster.

---

# 🎯 7. Hyperparameter Tuning (NN)

Explored combinations:

* Learning Rates: `[0.01, 0.001, 0.0001]`
* Batch Sizes: `[32, 64, 128]`


Best configuration:

```
Learning rate = 0.001
Batch size    = 128
R²            = 0.8174
```

---

# 💾 8. Model Saving

All models and preprocessing components were saved:

```
SVR_model.joblib
DecisionTree_model.joblib
RandomForest_model.joblib
KerasModel.keras

label_encoders.joblib
feature_order.joblib

standard_scaler_sklearn.joblib
standard_scaler_X_keras.joblib
standard_scaler_y_keras.joblib
```

This completes the deployment pipeline.

---

# 📈 9. Final Model Comparison

| Model             | MAE       | MSE       | R²        | Notes  |
| ----------------- | --------- | --------- | --------- | ------ |
| **Random Forest** | **64.06** | **31078** | **0.827** | ⭐ Best |
| Neural Network    | 76.53     | 32645     | 0.818     | Strong |
| Decision Tree     | 72.54     | 38164     | 0.787     | Good   |
| SVR               | 96.98     | 62853     | 0.650     | Weak   |

---

# 🌐 10. Deployment (Streamlit)

### Input → Preprocessing → Prediction

The Streamlit app:

* Loads encoders & scalers
* Preprocesses input automatically
* Produces predictions from **all 4 models**
* Displays results cleanly

Run with:

```bash
streamlit run streamlit_app.py
```

---

# 🚀 11. Conclusion

This project successfully:

* Built a complete ML pipeline
* Engineered meaningful features
* Trained 4 regression models
* Performed RF feature analysis
* Trained a Neural Network with GPU
* Tuned hyperparameters
* Saved deployable models
* Built a Streamlit prediction app

**Random Forest is the best-performing model.**
