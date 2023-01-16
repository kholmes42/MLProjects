
Ensemble learning combines a series of learners to make a final prediction. Typically weaker learners are used in ensembles (specifically decision trees/stumps) but any model can be used. This allows for ensembles to learn non-linear decision boundaries.

**Bagged Forest:** Use a sequence of decision trees to form an ensemble. Individual learners are trained using bootstrapped data (repeated sampling with replacement from training dataset). Final prediction is average of individual trees.

**Extra Trees Forest:** Use a sequence of decision trees to form an ensemble. Splits on the individual trees are performed randomly instead of greedily. This means training is much faster and the overall model has very low variance.

**Random Forest:** Use a sequence of decision trees to form an ensemble. Individual learners are trained on bootstrapped data and each tree only uses a random subset of all available features. This technique decorrelates individual trees reducing variance of overall prediction.

**Gradient Boosted Trees:** Use a sequence of decision trees to form an ensemble where each subsequent tree is trained to learn from errors made on previous trees. GB Trees typically perform well on tabular datasets and are often best performing blackbox model.
