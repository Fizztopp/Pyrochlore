import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl


def gauss(sigma, shift, x):
    return np.exp(-0.5*((x-shift)/sigma)**2)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['font.size'] = 20   # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['figure.figsize'] = [15.,5.]

#file_BZ = open('k_BZ_full.txt','r')
file_PARAMS = open('parameters.txt','r')


file_PARAMS.close()

# PLOT: MAGNETIZATION ############################################################################## 
file_m_160_050 = open('m_t_MF.txt','r')
file_E_160_050 = open('E_t.txt','r')


m_160_050 = np.loadtxt(file_m_160_050)
E_160_050 = np.loadtxt(file_E_160_050)

file_m_160_050.close()
file_E_160_050.close()

timesteps = 200000

t = np.linspace(-10., 190, timesteps-1)
tf = np.linspace(-10., 190, timesteps)


fig1 = plt.figure(2)
gs1 = gridspec.GridSpec(1, 2)
fig1.suptitle(r'Quench: $U=1.60$ $\to$ $U=1.45$ '+r', stepsize = 0.0005'+r', #BZ = 30*30*30')  

print(abs(E_160_050[1]-E_160_050[18])*0.8)

ax00 = fig1.add_subplot(gs1[0,0])
ax00.set_title(r"Magnetization")
ax00.plot(t,m_160_050, "b")
#ax00.axhline(y=0.24989, xmin=0.0, xmax = 1.0, c="b", linestyle="--", linewidth=1.0, label=r"$m_{initial}$ (EQ)")
#ax00.axhline(y=0.30608, xmin=0.0, xmax = 1.0, c="g", linestyle="--", linewidth=1.0, label=r"$m_{initial}$ (EQ)")
ax00.axvline(x=0.0, ymin=0.0, ymax = 1.0, c="k", linestyle="-")
#ax00.axhline(y=0.0, xmin=0.0, xmax = 1.0, c="k", linestyle="--", label=r"$m_{final}$ (EQ)")
#ax00.set_xlim(-10.,90.0)
ax00.set_ylim(0.00,0.35)
ax00.set_ylabel('m(t)')
ax00.set_xlabel('time t')

ax01 = fig1.add_subplot(gs1[0,1])
ax01.set_title(r"Energy")
ax01.plot(tf,E_160_050[:,0]*0.8, "r--", markevery=4000, label=r"$E_{tot}$ ($U_{init} = 1.60$)")
#ax01.plot(tf,E_160_050[:,1], "g--", markevery=4000, label=r"$E_{pot}$ ($U_{init} = 1.60$)")
#ax01.plot(tf,E_160_050[:,2], "b--", markevery=4000, label=r"$E_{kin}$ ($U_{init} = 1.60$)")
ax01.axvline(x=0.0, ymin=0.0, ymax = 1.0, c="k", linestyle="-")
#ax01.axhline(y=-0.0252, xmin=0.0, xmax = 100.0, c="k", linestyle="--", label=r"$m_{final}/F_{final}$ (stable EQ)")
ax01.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
#ax01.set_xlim(-9.9,89.9)
ax01.set_ylabel('E(t)')
ax01.set_xlabel('time t')

plt.tight_layout()
plt.show()