import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.33   # capital share
s = 0.20       # savings rate
delta = 0.05   # depreciation
n = 0.01       # population growth
g = 0.02       # technology growth
T = 100        # periods
k0 = 1.0       # initial capital per effective worker

def next_k(k):
    return (s * k**alpha + (1 - delta) * k) / ((1 + n) * (1 + g))

k = np.zeros(T)
k[0] = k0
for t in range(1, T):
    k[t] = next_k(k[t-1])

y = k**alpha
c = (1 - s) * y

print(f"Approx steady-state k*: {k[-1]:.4f}")

plt.figure()
plt.plot(k, label='Capital per eff. worker (k_t)')
plt.plot(y, label='Output per eff. worker (y_t)')
plt.title('Solow Model Convergence')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()
