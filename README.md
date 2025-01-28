# Remaining Useful Life (RUL) Prediction of Turbofan Engines
This project aims to predict the Remaining Useful Life (RUL) of turbofan engines using machine learning techniques, including Linear Regression, Random Forest, and XGBoost. The model is trained and evaluated using the NASA Turbofan Engine Dataset.
## Dataset

The dataset used in this project is the **NASA Turbofan Engine Dataset**, which consists of simulations of turbofan engine degradation. The dataset is divided into four subsets (FD001, FD002, FD003, and FD004), each representing different operational conditions and failure modes. Key aspects of the dataset include:

- **Challenges**: The dataset consists of 4 separate challenges with increasing complexity. Engines operate normally at first but develop failures over time.
- **Data**: For each engine, the data includes:
  - Engine unit number
  - Operation cycles
  - Three operational settings
  - Readings from 21 sensors
- **Objective**: The goal is to predict the RUL of each engine in the test set, where data is collected up to a point before failure.

### Summary Table of Challenges

| Dataset | Operational Conditions | Failure Modes | Training Size (number of engines) | Test Size (number of engines) |
|---------|------------------------|---------------|-----------------------------------|--------------------------------|
| FD001   | 1                      | 1             | 100                               | 100                            |
| FD002   | 6                      | 1             | 260                               | 259                            |
| FD003   | 1                      | 2             | 100                               | 100                            |
| FD004   | 6                      | 2             | 248                               | 249                            |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow/Keras for neural network models
- Matplotlib/Seaborn for visualization

## References
https://medium.com/@rohit.malhotra67/predictive-maintenance-on-nasas-turbofan-engine-degradation-dataset-cmapss-c066ee427931

data repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan
