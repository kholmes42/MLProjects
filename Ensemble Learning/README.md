
Ensemble learning combines a series of learners to make a final prediction. Typically weaker learners are used in ensembles (specifically decision trees/stumps) but any model can be used. This allows for ensembles to learn non-linear decision boundaries.

**Bagged Forest:** Use a sequence of decision trees to form an ensemble. Individual learners are trained using bootstrapped data (repeated sampling with replacement from training dataset). Final prediction is average of individual trees.

**Extra Trees Forest:** Use a sequence of decision trees to form an ensemble. Splits on the individual trees are performed randomly instead of greedily. This means training is much faster and the overall model has very low variance.

**Random Forest:** Use a sequence of decision trees to form an ensemble. Individual learners are trained on bootstrapped data and each tree only uses a random subset of all available features. This technique decorrelates individual trees reducing variance of overall prediction.

**Gradient Boosted Trees:** Use a sequence of decision trees to form an ensemble where each subsequent tree is trained to learn from errors made on previous trees. GB Trees typically perform well on tabular datasets and are often best performing blackbox model.


## End-to-End XGBoost Walkthrough
This end-to-end walkthrough covers multiple aspects of a data science problem, in the context of a binary classification problem using a XGBoost Classifier.

The problem involves classifying survey respondents into roles of either 'Software Engineer' or 'Data Scientist'. For this problem I use a 14 item featureset. The features contain a mix of string and numeric datatypes, and involves some preprocessing inorder to get it into a working format. This includes encoding categorical reponses as well as parsing string entries and converting them into a numeric form.

#### EDA:
To begin, I split the data into training and testing datasets, the training dataset will subsequently be used in a cross validation (CV) approach to model training. First we notice a slight class imbalance, but nothing too extreme.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/classeda.png)

I also examine the heatmap of feature correlations to spot any intuitive relationships. An inutitive positive relationship between age, experience and compensation shows up.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/featcorr.png)

#### Training/Testing:

I run a 5-fold GridSearch CV on the training data to train an XGBoost model with multiple parameters. Below is the learning curve for the model. This shows how the model training/validation error changes with training data size. It shows us that the validation error is improving as we add more data observations, but the rate of improvement slows down after 1500 observations. The gap between training and validation is not large which suggests that the training process has worked well.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/learncurve.png)

The below charts show the confusion matrix, precision/recall graph and AUC graph for the testing data. The metric used for evaluation is F1-score, and the final model has a score of about 0.71 on the testing data.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/metrics.png)

#### Model Interpretability:

Non-linear models such as ensemble based XGBoost are often called "black-box" models. They do not readily have available simple coefficients like a linear regression model does. However, a slate of techniques have been introduced to aid with model interpretability. These include Feature Importance, Partial Dependence Plots (PDP) and SHAP values. I examine them each in turn below.

Feature importance is a global metric that allows us to examine what features are having the largest impact on the final output of the model. This is done by measuring the impact that each feature has in the decision process throughout the ensemble. While they do not show the direction that the final values are impacted, intuitive rationale can be gleaned. In the context of classfying a software engineer versus a data scienctist response, R and Math appear at the top. It would be logical to suggest that data scientists are more mathematically inclined than software engineers and also more frequently use statistical programming languages such as R.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/featimp.png)

PDPs show the impact of utilizing the trained model but artificially altering a feature value in the dataset and holding others constant. Below we plot the PDP for Education, Age and Experience. The first plot suggests that as education increases, the outcome is more likely to be a data scientist. Frequently higher education is required to become a data scientist so this likely makese sense. In the second plot we do not see much of a relationship. Finally, in the third plot, we find that as experience increases, it is more likely that the person is a software engineer. It is harder to gather an explanation for this last one, however it is possible that data scientist is still a relatively new term so there are people with less experience with that job title.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/pdp.png)

SHAP values help gain understanding into the directional impact of feature values on the outcome. In the chart below, values to the right indicate the response is more likely to be a software engineer, while values to the left indicate the response is more likely to be a data scientist. Red indicates the value is relatively high compared to other values of the feature and blue indicates it is relatively low. As expected, people who indicated R experience and Math backgrounds are more likely to be classified as data scientists. From the below we also get a small indication that data scientists may have higher compensation. As we move to the bottom of the chart, the feature values are less dispersed and offer less of an impact on the final output classification.

![plot](https://github.com/kholmes42/MLProjects/blob/main/imgs/shap.png)
