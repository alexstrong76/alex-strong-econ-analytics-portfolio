import numpy as np

rho = -0.5                # (sigma-1)/sigma
p = np.array([2.0, 3.0])  # prices
I = 100.0                 # income

def ces_demand(p, I, rho, w=np.array([0.5, 0.5])):
    # symmetric two-good case
    P = (w[0]*p[0]**(1-rho) + w[1]*p[1]**(1-rho))**(1/(1-rho))
    C = I / P
    x = C * w * (p / P)**(-1/(1-rho))
    return x, P

x, P = ces_demand(p, I, rho)
print(f"CES price index P: {P:.4f}")
print("Optimal consumption bundle:", x)
