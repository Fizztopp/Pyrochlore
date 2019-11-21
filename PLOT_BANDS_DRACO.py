#Python programm to plot the plot the bands calculated by Pyrochlore.cc

import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec

file_BANDS = open('bands0.txt','r')
file_BZ = open('k_BZ_full.txt','r')
file_PARAMS = open('parameters.txt','r')
file_mu = open('mu.txt','r')

PARAMS = np.loadtxt(file_PARAMS)
mu        = np.round(PARAMS[0],3)
ts        = np.round(PARAMS[1],3)
U         = np.round(PARAMS[2],3)
U_quench  = np.round(PARAMS[3],3)
T_RAMP    = np.round(PARAMS[4],3)
beta_S    = np.round(PARAMS[5],3)
J1        = np.round(PARAMS[6],3)
beta_B    = np.round(PARAMS[7],3)
starttime = np.round(PARAMS[8],3)
endtime   = np.round(PARAMS[9],3)
timesteps = np.round(PARAMS[10],3)
stepsize  = np.round(PARAMS[11],3)
Omega_cut = np.round(PARAMS[12],3)   
SHIFT     = np.round(PARAMS[13],3)
DELAY     = np.round(PARAMS[14],3)
SIGMA     = np.round(PARAMS[15],3)

MAT_BANDS = np.loadtxt(file_BANDS)
MAT_BZ = np.loadtxt(file_BZ)
mu_t = np.loadtxt(file_mu)

file_BANDS.close()
file_BZ.close()
file_PARAMS.close()
file_mu.close()

N_BAND = np.size(MAT_BANDS[:,0])
N_BZ = np.size(MAT_BZ[:,0])
k=np.arange(N_BAND)     

#mu = 0.0    

fig1 = plt.figure(2, figsize=(8,8))
gs1 = gridspec.GridSpec(1, 1)
#fig1.suptitle(r'EQ band structure: $m = $'+str(M)+r', $t_\sigma = $'+str(ts)+r',  $U = $'+str(U)+r',  $\beta_S = $'+str(beta_S)+r', #BZ = 30x30x30' ,fontsize=24)

ax11 = fig1.add_subplot(gs1[0,0])
#ax11.set_xticks([0 , 100, 200, 300, 400, 500, 600])
#ax11.set_xticklabels(['$\Gamma$', 'X', 'W', 'L', '$\Gamma$', 'K', 'X'],fontsize=24)
ax11.set_ylabel('E(k), U=1.60',fontsize=24)
ax11.plot(k,MAT_BANDS[:,0], 'b', linewidth=2.0, label=r"band structure")
ax11.plot(k,MAT_BANDS[:,1], 'b', linewidth=2.0)
ax11.plot(k,MAT_BANDS[:,2], 'b', linewidth=2.0)
ax11.plot(k,MAT_BANDS[:,3], 'b', linewidth=2.0)
ax11.plot(k,MAT_BANDS[:,4], 'b', linewidth=2.0)
ax11.plot(k,MAT_BANDS[:,5], 'b', linewidth=2.0)
ax11.plot(k,MAT_BANDS[:,6], 'b', linewidth=2.0)
ax11.plot(k,MAT_BANDS[:,7], 'b', linewidth=2.0)
ax11.plot(k, [mu]*np.size(k), 'k--', label=r"$\mu_{intial}$")
ax11.set_xlim(0,N_BAND)
#ax11.legend(fontsize=24)
plt.grid(True)

plt.show()