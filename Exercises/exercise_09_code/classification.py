import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import math


from models import LinearModel, LogisticModel


def root_mean_squared_error(y, y_hat):
    res = math.sqrt(np.mean((y-y_hat)**2))
    # <Exercise 9.1.1>
    return res


def calculate_metrics(y, y_hat):
    # <Exercise 9.1.3>
    TN, FP, FN, TP = confusion_matrix(y, y_hat).ravel()
    precision      = TP/(TP+FP)
    recall         = TP/(TP+FN)
    
    # <Exercise 9.1.4>
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    
    return {
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy
    }


# Define the number of samples to generate your dataset.
samples = 1000

# Generate the (ground truth) dataset as a straight line with some
# Gaussian noise.
np.random.seed(0)

X = np.random.normal(size=samples)
y = (X > 0).astype(np.float64)
X[X > 0] *= 4
X += .3 * np.random.normal(size=samples)
X = X[:, np.newaxis]

# add noise the y values.
jitter_y = y.copy().astype(np.float64)
np.random.seed(len(jitter_y))
jitter_y = jitter_y + 0.075 * np.random.rand(len(jitter_y)) - 0.05

# Define the test variable.
min_value, max_value = X.min(), X.max()
values = np.linspace(np.min(X), np.max(X), 1000).reshape((-1, 1))

# Linear regression.
linear = LinearModel()
linear.fit(X, y)

# Prediction.
y_predict = linear.predict(X)

# Plot the classification data.
plt.subplot(2, 1, 1)
plt.title("Linear Regression")
plt.xlim([min_value - 0.25, max_value + 0.25])
plt.ylim([-0.8, +1.8])

y_boolean = y_predict > 0.5
linear_predictions = y_boolean
plt.scatter(np.squeeze(X[~y_boolean]), jitter_y[~y_boolean], 30, alpha=0.35)
plt.scatter(np.squeeze(X[y_boolean]), jitter_y[y_boolean], 30, alpha=0.35)

# Plot the test variable prediction.
y_predict = linear.predict(values)
plt.plot(values, y_predict, "C2")

# Plot the decision boundary.
boundary = (0.5 - linear.intercept_) / linear.coef_
plt.plot((boundary, boundary), (-0.75, 1.75), "C3--")
plt.legend(["Linear Regression", "Decision Boundary", "Class 1", "Class 2"])
plt.axhline(.5, color=".8")

# Logistic regression.
logistic = LogisticModel()  # linear_model.LogisticRegression(C=1e5)

# y = np.int64(y)
logistic.fit(X, y)

# Prediction.
y_predict = logistic.predict(X).ravel()

# Plot the logistic regression.
plt.subplot(2, 1, 2)
plt.title("Logistic Regression")
plt.xlim([min_value - 0.25, max_value + 0.25])
plt.ylim([-0.8, +1.8])

y_boolean = y_predict > 0.5
logistic_predictions = y_boolean
plt.scatter(np.squeeze(X[~y_boolean]), jitter_y[~y_boolean], 30, alpha=0.35)
plt.scatter(np.squeeze(X[y_boolean]), jitter_y[y_boolean], 30, alpha=0.35)

# Plot the test variable prediction.
y_predict = logistic.predict(values)
plt.plot(values, y_predict, "C2")

# Plot the decision boundary.
boundary = (0.0 - logistic.intercept_) / logistic.coef_
plt.plot((boundary[0], boundary[0]), (-0.75, 1.75), "C3--")
plt.legend(["Logistic Regression", "Decision Boundary", "Class 1", "Class 2"])
plt.axhline(.5, color=".8")

# <Exercise 9.1.2>
RMSE_linear = root_mean_squared_error(y, linear_predictions)
RMSE_logistic = root_mean_squared_error(y, logistic_predictions)
print(RMSE_linear)
print(RMSE_logistic)

# Variables:
# y: the ground truth values
# linear_predictions: predicted y-values using linear regression
# logistic_predictions: predicted y-values using logistic regression

# <Exercise 9.1.5>
matrics_linear = calculate_metrics(y, linear_predictions)
matrics_logistic = calculate_metrics(y, logistic_predictions)
print(matrics_linear)
print(matrics_logistic)

# Show the matplotlib window.
plt.show()
