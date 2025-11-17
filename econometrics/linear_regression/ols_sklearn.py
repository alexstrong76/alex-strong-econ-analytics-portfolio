import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(7)
n = 500
X = np.random.randn(n, 3)
beta_true = np.array([1.5, -0.7, 0.3])
y = X @ beta_true + np.random.randn(n) * 0.6

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=7)
model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 (test):", r2_score(yte, pred))
