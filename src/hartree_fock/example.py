import numpy as np

n = 20
m = 40

F = np.reshape(np.arange(m*n*n), (m,n,n))
G1 = np.zeros((n,n,n,n))


G = np.einsum('Fab,Fcd->bacd',
              F,
              F)

print("Finalizado.")

for a in range(n):
    for b in range(n):
        for c in range(n):
            for d in range(n):
                for f in range(m):
                    G1[b,a,c,d] += F[f,a,b]*F[f,c,d]

print("Finalizado 2.")
print(np.allclose(G, G1))
