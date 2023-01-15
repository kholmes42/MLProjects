
## Data Prep
Many ML models work better when features are on same scale. Technique's used involve standardization and scaling. Important to not let leakage of testing data impact data prep stage. Also requires decisions on what to do with missing data.

**Scaling:** Involves bounding/converting feature values to between predifined ranges (only can use if these naturally exist). For example map values to between 0 and 1.

**Standardization:** Convert feature values to standard normal distribution (mean of 0 and unit variance).

**Missing Data:** Depending on amount of missing data, can impute (use regression or average of feature value to fill in missing) if not too many, drop if way too much is missing or very small amount is missing.


## Outliers & Novelty

Outlier datapoints involve anomalous data in training phase. Certain models can be severly impacted by outliers in data which will impact model performance. Run algorithms to detect these points prior to training, then work to understand what is causing them (i.e. false data, true data, one off, possible repeated?) and what to do about them (drop the datapoints, winsorize the datapoints, etc.).

Novelty datapoints involve anomalous data coming in during testing or production phases of the model. Must determine if they the model is appropriate to be run on them if they were never really encountered during training.

**Local Outlier Factor:** Run a KNN to determine comparison in distance between data points and the average distance of all data points. Data points that have far average distance to all data points are outliers.

**Isolation Forest:** Random Forest implementation of random splits on features for each data point, if it only takes a few splits to put a sample data point in a leaf node then that means it is easy to separate it from the dataset (potentially an outlier).

**Elliptic Envelope:** Fit Gaussian distribution to dataset using robust covariance, data points in tails of distribution can be considered outliers (assumes data is normally distributed).






