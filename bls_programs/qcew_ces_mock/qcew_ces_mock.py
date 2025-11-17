import numpy as np
import pandas as pd

np.random.seed(77)
industries = ['NAICS-54','NAICS-62','NAICS-44','NAICS-92']
quarters = pd.period_range('2019Q1','2025Q4', freq='Q')
rows = []
for ind in industries:
    level = 1000 + np.random.randint(-100,100)
    for q in quarters:
        growth = np.random.normal(0.0, 0.02)
        level = max(100, level*(1+growth))
        rows.append((ind, str(q), int(level)))
df = pd.DataFrame(rows, columns=['industry','quarter','employment'])

df['employment_lag4'] = df.groupby('industry')['employment'].shift(4)
df['employment_lag1'] = df.groupby('industry')['employment'].shift(1)
df['YoY_%'] = 100*(df['employment']/df['employment_lag4'] - 1)
df['QoQ_%'] = 100*(df['employment']/df['employment_lag1'] - 1)
print(df.head(12))
