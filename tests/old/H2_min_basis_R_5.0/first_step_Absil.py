import numpy as np

##For H2 at R=5.0
Ua = np.array([0.73264959, -0.6806056]).reshape((2,1))
Ub = np.array([0.73264959,  0.6806056]).reshape((2,1))
C0 = 0.756853272254
C1 = -0.653584825618

eta_a = np.array([-11.59165006, -8.02491125]).reshape((2,1))
eta_b = np.array([-11.59165006,  8.02491125]).reshape((2,1))

print('Ua:\n', Ua)
print('Ub:\n', Ub)

print('eta_a:\n', eta_a)
print('eta_b:\n', eta_b)

Proj_a = np.identity(2) - Ua @ Ua.T
Proj_b = np.identity(2) - Ub @ Ub.T

print('Proj_a:\n', Proj_a)
print('Proj_b:\n', Proj_b)

f = C0 * Ua[0,0] * Ub[0,0] + C1 * Ua[1,0] * Ub[1,0]
print("f = ", f)

grad_a = np.zeros((2,1))
grad_b = np.zeros((2,1))

grad_a[0,0] = C0 * Ub[0,0]
grad_a[1,0] = C1 * Ub[1,0]

grad_b[0,0] = C0 * Ua[0,0]
grad_b[1,0] = C1 * Ua[1,0]

print('grad_a:\n', grad_a)
print('grad_b:\n', grad_b)

print('Proj_a grad_a:\n', Proj_a @ grad_a)
print('Proj_b grad_b:\n', Proj_b @ grad_b)

hess_a = np.diag((C0, C1)) @ eta_b
hess_b = np.diag((C0, C1)) @ eta_a

print('hess_a:\n', hess_a)
print('hess_b:\n', hess_b)

print('LHS_a (without the projector):\n', eta_a @ Ua.T @ grad_a - hess_a)
print('LHS_b (without the projector):\n', eta_b @ Ub.T @ grad_b - hess_b)

print('LHS_a:\n', Proj_a @ eta_a @ Ua.T @ grad_a - Proj_a @ hess_a)
print('LHS_b:\n', Proj_b @ eta_b @ Ub.T @ grad_b - Proj_b @ hess_b)



