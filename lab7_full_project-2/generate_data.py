import numpy as np

for N in [100, 1000, 10000]:
    np.random.seed(42)
    a = np.random.randn(N-1)
    c = np.random.randn(N-1)
    b = np.zeros(N)
    b[0] = abs(c[0]) + 2
    for i in range(1, N-1):
        b[i] = abs(a[i-1]) + abs(c[i]) + 2
    b[N-1] = abs(a[N-2]) + 2
    
    x_true = np.random.randn(N)
    d = np.zeros(N)
    d[0] = b[0]*x_true[0] + c[0]*x_true[1]
    for i in range(1, N-1):
        d[i] = a[i-1]*x_true[i-1] + b[i]*x_true[i] + c[i]*x_true[i+1]
    d[N-1] = a[N-2]*x_true[N-2] + b[N-1]*x_true[N-1]
    
    np.savez(f'system_{N}.npz', a=a, b=b, c=c, d=d, x_true=x_true)
    print(f'Generated N={N}')
