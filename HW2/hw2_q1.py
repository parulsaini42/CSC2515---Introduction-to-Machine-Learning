import numpy as np
import matplotlib.pyplot as plt
sigma_sq = 9
mu = 1
n = 10
lam=  [0.01, 0.02, 0.04, 0.08, 0.16, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
squared_error = []
bias = []
variance = []
for i in range(len(lam)):
    sq =   ((mu * lam[i])/(1 + lam[i]))**2 + ( sigma_sq / (n * (1 + lam[i])**2))
    b = ((mu * lam[i])/(1 + lam[i]))**2
    var = sigma_sq / (n * (1 + lam[i])**2 )
    squared_error.append(sq)
    bias.append(b)
    variance.append(var)

plt.plot(lam ,squared_error, color='red',label='Squared Error')
plt.plot(lam, bias, color='green',label='(Bias)^2')
plt.plot(lam,variance, color='blue',label='Variance')
plt.xlabel('λ')
plt.legend()
plt.show()

 