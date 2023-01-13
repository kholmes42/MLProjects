ML modelling process involves two key aspects. Choosing evaluation criteria, and then picking the right model such that it balances overfitting and underfitting on out of sample data.

**Evaluation Metrics:**
Regression: MSE, RMSE, MAE, MAPE, R^2
Classfication: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix

**Model Selection:**
If only simple model being trained -> Train Test Split
If multiple models or hyperparameter tuning required -> Cross validation techniques

In time series domain, ensure walkforward CV is used to prevent leakage. When dealing with small dataset can use Leave-one-out CV.

