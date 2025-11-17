import numpy as np
import pandas as pd

np.random.seed(42)
n = 500
df = pd.DataFrame({
    'emp_id': np.random.randint(1000, 1100, n),
    'week': np.random.randint(1, 10, n),
    'hourly_rate': np.random.choice([15,18,22,28], n, p=[0.3,0.3,0.25,0.15]),
    'hours': np.clip(np.random.normal(42, 6, n), 20, 70),
    'rounded_to': np.random.choice([1, 6, 15], n, p=[0.2, 0.5, 0.3])
})

# Overtime (simple example, 1.5x over 40)
df['ot_hours'] = (df['hours'] - 40).clip(lower=0)
df['reg_hours'] = df['hours'] - df['ot_hours']
df['reg_pay'] = df['reg_hours'] * df['hourly_rate']
df['ot_pay'] = df['ot_hours'] * df['hourly_rate'] * 1.5
df['gross'] = df['reg_pay'] + df['ot_pay']

# Simple flags an analyst might explore
df['potential_rounding_issue'] = df['rounded_to'].ge(15)
df['long_week_flag'] = df['hours'].ge(55)

print(df.head())
print("\nAggregate preview:")
print(df[['gross','ot_hours','potential_rounding_issue','long_week_flag']].describe(include='all'))
