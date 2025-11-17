import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(0)
T = 120
eps = np.random.randn(T) * 0.5
g = 0.5 + 0.6 * np.roll(eps, 1)
g[0] = eps[0]  # quarterly "growth" points (toy)
ts = pd.Series(g)

model = ARIMA(ts.astype(float), order=(1,0,1)).fit()
print(model.summary())
