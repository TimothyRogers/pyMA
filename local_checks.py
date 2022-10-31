#%%

import numpy as np
from matplotlib import pyplot as plt

from pyma import systems
from pyma import fdd

#%%

M = np.eye(2)
K = np.array([[20, -10],[-10, 20]])
C = 0.05*M + 0.001*K
MI = np.linalg.inv(M)

model = systems.SpatialModel(M=M,C=C,K=K)

A = model.first_order_form()
B = np.block([[np.zeros_like(M)],[np.linalg.inv(M)]])
C = np.hstack((-MI@K,-MI@C))
D = MI

dt = 1e-1 # 10Hz sample freq

# Deterministic SSM
ssm = systems.StateSpace(A=A, B=B, C=C, D=D, Q=np.kron(np.array([[0,0],[0,1]]),np.identity(2)),R=np.identity(2), dt=dt)
ssm.discretise()
x, y = ssm.simulate(T=2000, x0=np.zeros((A.shape[0],)))

plt.figure()
plt.plot(y.T)
plt.show(block=False)

# opts = fdd.SVS.opts
opts = {'nfft':512, 'noverlap':500}
SVS = fdd.SVS(opts)(y[:,1:].T)

plt.figure()
plt.semilogy(SVS[0])
plt.show(block=True)

