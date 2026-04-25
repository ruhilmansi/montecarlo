monte carlo integration by implementing both serial and parallel versions using multiprocessing and studying how sample size and parallelization affect accuracy and execution time

for importance sampling, used truncated exponential

CDF of g(x) is G(x) = (1 - e^-x) / (1 - e^-2)

inverse G^-1(u) = -ln(1 - u * (1 - e^-2))

in code montecarlo.py

```python
norm = 1 - np.exp(-2)       
u = np.random.uniform(0, 1)  
x = -np.log(1 - u * norm)    
```
