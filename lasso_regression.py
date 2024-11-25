import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Define a function that allows one to quickly calculate the Root Mean Squared Error
def RMSE(y: np.array, y_hat: np.array) -> float:
    return np.sqrt(np.sum((y - y_hat) ** 2) / len(y))

# Define a function that allows one to quickly calculate R Square
def R2(y: np.array, y_hat: np.array) -> float:
    SSR = np.sum(1/len(y)*(y_hat - y)**2)
    SST = np.sum(1/len(y)*(np.mean(y)-y)**2)
    R_sq = 1 - SSR / SST
    return R_sq

# Load the data which is stored in numpy arrays
x = np.load(r'/Users/evan/UoE/Year 2/Facets/Regression Modelling/End of Theme 1/rm_eot_x.npy')

y = np.load(r'/Users/evan/UoE/Year 2/Facets/Regression Modelling/End of Theme 1/rm_eot_y.npy')

"""
Lasso Regression via Gradient Descent
"""

class LassoRegression:
    def __init__(self, step_size: float, iterations: int, l1_penalty: float):
        self.step_size = step_size
        self.iterations = iterations
        self.l1_penalty = l1_penalty

        # b_hat are the coefficients of the predictors
        self.b_hat = None
        # bias, also known as the intercept term
        self.intercept = None


    def fit(self, x: np.ndarray, y: np.array) -> None:
        if len(x.shape) == 2:
            num_samples, num_predictors = x.shape
        else:
            num_samples, num_predictors = len(x), 1

        self.b_hat = np.zeros(num_predictors)
        self.intercept = 0

        for i in range(self.iterations):
            y_hat = self.predict(x)
            # Function to minimise is given by:
            #    (RSS(x) + t * sum(abs(b_hat)) ) / m
            #  = (sum((y - y_hat)^2) + t * sum(abs(b_hat))) / m
            self._update_b_hat(x, y, y_hat, num_samples)
            self._update_intercept(y, y_hat, num_samples)

    # Partial derivative with respect to b_hat is given by:
    # When b_hat >= 0:
    #     -2 * (x dot sum(y - y_hat)) + l1_penalty / m
    # When b_hat < 0:
    #     -2 * (x dot sum(y - y_hat)) - l1_penalty / m
    #
    # Combines to:
    #     -2 * (x dot sum(y - y_hat)) + sign(b_hat) * l1_penalty / m
    def _update_b_hat(self, x: np.ndarray, y: np.array, y_hat: np.array, num_samples: int) -> None:
        d_b_hat = (-2 / num_samples) * np.dot(x.T, (y - y_hat)) + np.sign(self.b_hat) * self.l1_penalty
        self.b_hat -= self.step_size * d_b_hat

    # Partial derivative with respect to intercept is given by:
    #     -2 * (sum(y - y_hat)) / m
    def _update_intercept(self, y: np.array, y_hat: np.array, num_samples: int) -> None:
        d_bias = (-2 / num_samples) * np.sum(y - y_hat)
        self.intercept -= self.step_size * d_bias

    def predict(self, x: np.ndarray) -> float:
        return self.intercept + np.dot(x, self.b_hat)


def main():
    # Importing dataset
    df = pd.read_csv('Student_Performance.csv', sep=',', header=0)
    df.replace({"Yes": 1, "No": 0}, inplace=True)


    df = df.to_numpy()

    # X is our given predictors
    x = df[:, :-1]
    # Y is our given observations
    y = df[:, -1]

    # Standardize features
    x = StandardScaler().fit_transform(x)

    # Splitting the data into training data and test data with an 80% and 20% split
    num_rows, num_predictors = x.shape

    x_train = x[: int(num_rows * 0.8)]
    y_train = y[: int(num_rows * 0.8)]
    x_test = x[int(num_rows * 0.8) :]
    y_test = y[int(num_rows * 0.8) :]

    # Applying gradient descent over 1000 iterations, with step_size 0.01 and l1_penalty (t) as 1.
    lasso = LassoRegression(step_size=0.01, iterations=1000, l1_penalty=1)
    lasso.fit(x_train, y_train)

    # Prediction on test set
    y_hat = lasso.predict(x_test)
    print(f"Predicted values: {y_hat[:5].round(2)}")
    print(f"Real values:      {y_test[:5]}")
    print(f"b_hat:        {lasso.b_hat.round(2)}")
    print(f"intercept:        {round(lasso.intercept, 2)}")

    print("RMSE: ", RMSE(y_test, y_hat))
    print("R2: ", R2(y_test, y_hat))

main()