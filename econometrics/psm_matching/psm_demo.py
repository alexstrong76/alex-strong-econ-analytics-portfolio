import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

np.random.seed(5)
n = 1500
x1 = np.random.randn(n)
x2 = np.random.randn(n)
ps = 1/(1+np.exp(-(0.5*x1 + 0.8*x2)))
t = np.random.binomial(1, ps)
tau = 1.5
eps = np.random.randn(n)*0.5
y = 3 + 1.0*x1 + 0.5*x2 + tau*t + eps

df = pd.DataFrame({"y":y,"t":t,"x1":x1,"x2":x2})

# Propensity score model
logit = LogisticRegression(solver="lbfgs").fit(df[["x1","x2"]], df["t"])
p_hat = logit.predict_proba(df[["x1","x2"]])[:,1]
df["ps"] = p_hat

# Match treated to control by ps
treated = df[df.t==1][["ps","y"]].reset_index()
control = df[df.t==0][["ps","y"]].reset_index()
nbrs = NearestNeighbors(n_neighbors=1).fit(control[["ps"]])
dist, idx = nbrs.kneighbors(treated[["ps"]])
att = (treated["y"].values - control.loc[idx.flatten(),"y"].values).mean()
print("ATT (nearest-neighbor on ps):", att)
