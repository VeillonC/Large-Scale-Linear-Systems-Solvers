import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from src.solvers import jacobi_dense, jacobi_sparse, gauss_seidel, sor_solver
from src.utils import generate_tridiagonal_matrix, compute_optimal_omega

def run_benchmarks():
    print("=== Large Scale Linear Systems - Solvers Benchmark ===\n")
    
    # 1. Comparison Dense vs Sparse Jacobi Execution Time
    sizes = [100, 200, 300, 400, 500]
    dense_times = []
    sparse_times = []
    
    print("Running Dense vs Sparse Jacobi performance tests...")
    for n in sizes:
        A = generate_tridiagonal_matrix(n)
        b = np.ones(n)
        A_csr = sp.csr_matrix(A)
        
        # Dense
        t0 = time.time()
        _, _, _ = jacobi_dense(A, b, max_iter=300)
        dense_times.append(time.time() - t0)
        
        # Sparse
        t0 = time.time()
        _, _, _ = jacobi_sparse(A_csr, b, max_iter=300)
        sparse_times.append(time.time() - t0)
        
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, dense_times, 'o-', label='Dense Jacobi', color='red')
    plt.plot(sizes, sparse_times, 's-', label='Sparse Jacobi (CSR)', color='green')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time: Dense vs Sparse Storage (Jacobi Solver)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/jacobi_dense_sparse.png')
    plt.close()
    print("Saved plot to assets/jacobi_dense_sparse.png")

    # 2. Optimal Omega Search for SOR Method
    n = 100
    A = generate_tridiagonal_matrix(n)
    b = np.ones(n)
    omega_opt_theory, rho_J = compute_optimal_omega(A)
    
    print(f"\nMatrix Size: {n}x{n}")
    print(f"Jacobi Spectral Radius: {rho_J:.4f}")
    print(f"Theoretical Optimal Omega: {omega_opt_theory:.4f}")
    
    omegas = np.linspace(1.0, 1.95, 20)
    iterations = []
    
    for w in omegas:
        _, iters, _ = sor_solver(A, b, omega=w, tol=1e-5, max_iter=2000)
        iterations.append(iters)
        
    plt.figure(figsize=(8, 5))
    plt.plot(omegas, iterations, 'b-o', label='SOR Iterations')
    plt.axvline(x=omega_opt_theory, color='r', linestyle='--', label=f'Theoretical $\\omega_{{opt}}$ ({omega_opt_theory:.3f})')
    plt.xlabel('Relaxation Parameter ($\\omega$)')
    plt.ylabel('Iterations to Convergence')
    plt.title('SOR Method Convergence vs Relaxation Parameter ($\\omega$)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('assets/omega_optimization.png')
    plt.close()
    print("Saved plot to assets/omega_optimization.png\n")

if __name__ == '__main__':
    run_benchmarks()
