ML modelling process involves two key aspects. Choosing evaluation criteria that is meaningful to the problem, and then picking the right model such that the metric balances overfitting and underfitting on out of sample data.

**Evaluation Metrics:**
Regression: MSE, RMSE, MAE, MAPE, Adj R^2, AIC, BIC
Classfication: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix

**Model Selection:**
If only simple model being trained -> Train Test Split
If multiple models or hyperparameter tuning required -> Cross validation techniques (watch for imbalanced classes)

**Additional Notes:** In time series domain, ensure walkforward CV is used to prevent data leakage. When dealing with small dataset can use Leave-one-out CV.

