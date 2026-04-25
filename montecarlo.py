import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
from multiprocessing import Pool, cpu_count
import sys

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

def target_func(x):
    return np.exp(-x) / (1 + x**2)


def mc_worker(args):
    method, n, bounds, seed = args
    np.random.seed(seed)
    a, b = bounds
    
    if method == 'standard':
        x = np.random.uniform(a, b, n)
        return np.sum(target_func(x))
    
    elif method == 'stratified':
        edges = np.linspace(a, b, n + 1)
        x = np.random.uniform(edges[:-1], edges[1:])
        return np.sum(target_func(x))
    
    elif method == 'importance':
        u = np.random.uniform(0, 1, n)
        norm = 1 - np.exp(-2)
        x = -np.log(1 - u * norm)
        weights = target_func(x) / (np.exp(-x) / norm)
        return np.sum(weights)

def run_parallel_mc(method, total_n, bounds, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    n_per_worker = [total_n // num_workers] * num_workers
    n_per_worker[-1] += total_n % num_workers
    
    seeds = np.random.randint(0, 10**8, num_workers)
    tasks = [(method, n_per_worker[i], bounds, seeds[i]) for i in range(num_workers)]
    
    start_time = time.time()
    with Pool(num_workers) as pool:
        sums = pool.map(mc_worker, tasks)
    end_time = time.time()
    
    total_sum = sum(sums)
    if method == 'importance':
        estimate = total_sum / total_n
    else:
        estimate = (total_sum / total_n) * (bounds[1] - bounds[0])
        
    return estimate, end_time - start_time


def main():
    print(" -- variance reduced parallel monte carlo integration -- ")
    
    bounds = (0, 2)
    exact_val, abs_err = integrate.quad(target_func, bounds[0], bounds[1])
    print(f"goal: integrate f(x) = e^-x / (1 + x^2) from {bounds[0]} to {bounds[1]}")
    print(f"exact value (scipy): {exact_val:.10f}\n")
    
    sample_sizes = [1000, 10000, 100000, 1000000]
    methods = ['standard', 'stratified', 'importance']
    results = {m: [] for m in methods}
    runtimes = {m: [] for m in methods}
    errors = {m: [] for m in methods}
    
    num_cpus = cpu_count()
    print(f"running on {num_cpus} CPU cores\n")
    
    print(f"{'method':<15} | {'samples':<10} | {'estimate':<15} | {'error':<15} | {'time (s)':<10}")
    
    for n in sample_sizes:
        for m in methods:
            est, duration = run_parallel_mc(m, n, bounds, num_workers=num_cpus)
            err = abs(est - exact_val)
            
            results[m].append(est)
            runtimes[m].append(duration)
            errors[m].append(err)
            
            print(f"{m.capitalize():<15} | {n:<10} | {est:<15.8f} | {err:<15.8e} | {duration:<10.4f}")
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'standard': '#FF6B6B', 'stratified': '#4D96FF', 'importance': '#6BCB77'}
    
    for m in methods:
        ax1.loglog(sample_sizes, errors[m], marker='o', label=m.capitalize(), color=colors[m], linewidth=2)
    
    ax1.set_title('error convergence', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('no of samples', fontsize=12)
    ax1.set_ylabel('absolute error', fontsize=12)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    for m in methods:
        ax2.scatter(runtimes[m], errors[m], label=m.capitalize(), color=colors[m], s=100, edgecolors='white', alpha=0.8)
        ax2.plot(runtimes[m], errors[m], color=colors[m], linestyle='--', alpha=0.4)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('accuracy vs runtime', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('execution time (sec)', fontsize=12)
    ax2.set_ylabel('absolute error', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = 'results.png'
    plt.savefig(plot_path, dpi=150)
    
    print(f"1. stratified sampling reduced error by approx {errors['standard'][-1]/errors['stratified'][-1]:.1f}x compared to standard")
    print(f"2. importance sampling reduced error by approx {errors['standard'][-1]/errors['importance'][-1]:.1f}x compared to standard")
    print(f"3. parallel processing distributed computation across all available cores")

if __name__ == '__main__':
    main()