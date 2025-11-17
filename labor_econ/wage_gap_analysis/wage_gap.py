import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(123)
n = 2000
years_exp = np.random.randint(0, 30, n)
edu = np.random.choice([12, 14, 16, 18], size=n, p=[0.2, 0.3, 0.4, 0.1])
gender = np.random.choice([0,1], size=n)  # 0=female, 1=male
industry = np.random.choice(['tech','health','retail','public'], size=n)

base = 15 + 1.2*years_exp + 2.0*(edu-12) + 3.0*gender
industry_prem = {'tech':7,'health':4,'retail':1,'public':2}
noise = np.random.randn(n)*3
wage = base + pd.Series(industry).map(industry_prem).values + noise

df = pd.DataFrame({'wage':wage,'years_exp':years_exp,'edu':edu,'gender':gender,'industry':industry})
X = pd.get_dummies(df[['years_exp','edu','gender','industry']], drop_first=True)
X = sm.add_constant(X)
model = sm.OLS(df['wage'], X).fit()
print(model.summary())
