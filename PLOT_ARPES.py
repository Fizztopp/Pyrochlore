import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 20   # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.figsize'] = [5.,10.]

file_PARAMS = open('parameters.txt','r')
file_BANDS0 = open('bands0.txt','r')
#file_BANDSp = open('bandsp.txt','r')
file_ARPES = open('ARPES.txt','r')
file_BZ = open('k_BZ_full.txt','r')
#file_EDC = open('EDC.txt','r')

PARAMS = np.loadtxt(file_PARAMS)
MAT_BANDS0 = np.loadtxt(file_BANDS0)
#MAT_BANDSp = np.loadtxt(file_BANDSp)
ARPES = np.loadtxt(file_ARPES)
MAT_BZ = np.loadtxt(file_BZ)
#EDC = np.loadtxt(file_EDC)

file_PARAMS.close()
file_BANDS0.close()
#file_BANDSp.close()
file_ARPES.close()
file_BZ.close()


mu = PARAMS[0]
omega_min = -0.5
omega_max = 0.5
k_min = 300
k_max = 400
N_BAND = np.size(MAT_BANDS0[:,0])
N_BZ = np.size(MAT_BZ[:,0])
k=np.arange(N_BAND)  
print(N_BAND)

fig1 = plt.figure(1, figsize=(15.,7.5))
#fig1.suptitle('TrARPES signal for Quench: $U=1.60$ (AFI) to $U=1.50$ (WSM) [30x30x30]')
gs1 = gridspec.GridSpec(1, 1)
ax00 = fig1.add_subplot(gs1[0,0])
#ax00.set_title('TrARPES signal for quench: $U=1.70$ (AFI) to $U=1.45$ (PMM) [30x30x30]')
#ax00.set_xticks([100 , 200, , 120, 160, 200, 240])
#ax00.set_xticklabels(['$\Gamma$', 'X', 'W', 'L', '$\Gamma$', 'K', 'X'])

MAX = np.amax(ARPES)

HEX=ax00.imshow(ARPES, aspect='auto', extent=[k_min,k_max,omega_min,omega_max], vmin=0.0, vmax=2.0, cmap="Reds", label=r"Tr-ARPES")
plt.colorbar(HEX)

ax00.plot(k,MAT_BANDS0[:,0], 'k', label=r"inst. eigen basis")
ax00.plot(k,MAT_BANDS0[:,1], 'k')
ax00.plot(k,MAT_BANDS0[:,2], 'k')
ax00.plot(k,MAT_BANDS0[:,3], 'k')
ax00.plot(k,MAT_BANDS0[:,4], 'k')
ax00.plot(k,MAT_BANDS0[:,5], 'k')
ax00.plot(k,MAT_BANDS0[:,6], 'k')
ax00.plot(k,MAT_BANDS0[:,7], 'k')
#ax00.plot(k, [mu]*N_BAND, 'k--', label=r"chemical potential")

ax00.set_xlabel('momentum')
ax00.set_ylabel('frequency')
ax00.legend()
ax00.set_ylim(omega_min,omega_max)
ax00.set_xlim(k_min,k_max)

#OMEGA = np.linspace(omega_min,omega_max,100)

#ax10 = fig1.add_subplot(gs1[1,0])
#ax10.plot(OMEGA, EDC/np.amax(EDC), 'k--', label=r"EDC @k=140")
#ax10.legend()

plt.show()  

