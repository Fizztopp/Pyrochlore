# Python programm to calculate the irreducible Brilluoin zone and the high symmetry path through the BZ for the Pyrochlore model

import numpy as np  
from scipy.linalg import expm, norm               
import spglib


k_num = 100
mesh = [30,30,30] 
a = 4.0 


# ROTATIONS
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

vec_rotx = np.array([1,0,0])
vec_roty = np.array([0,1,0])
vec_rotz = np.array([0,0,1])

ax = np.array([0,0,1])
M_rotz = M(ax,0)


def k_path():    
    '''
    Calculates high symmetry path points and saves as k_path.npz, #of points =  6*PC.k_num+1     
    '''
    MAT = np.zeros((3,3))                                
    MAT[:,0] = np.array([1,0,0])
    MAT[:,1] = np.array([0,1,0])
    MAT[:,2] = np.array([0,0,1])
    K_PATH = np.array([0,0,0])
    for GX in range(k_num):
        K_PATH = np.append(K_PATH, K_PATH[-3:]+1/k_num*2*np.pi/a*MAT[:,1])
    for XW in range(k_num):
        K_PATH = np.append(K_PATH, K_PATH[-3:]+1/k_num*np.pi/a*MAT[:,0])    
    for WL in range(k_num):
        K_PATH = np.append(K_PATH, K_PATH[-3:]-1/k_num*np.pi/a*MAT[:,1]+1/k_num*np.pi/a*MAT[:,2])  
    for LG in range(k_num):
        K_PATH = np.append(K_PATH, K_PATH[-3:]-1/k_num*np.pi/a*MAT[:,0]-1/k_num*np.pi/a*MAT[:,1]-1/k_num*np.pi/a*MAT[:,2])  
    for GK in range(k_num):
        K_PATH = np.append(K_PATH, K_PATH[-3:]+1/k_num*3*np.pi/(2*a)*MAT[:,0]+1/k_num*3*np.pi/(2*a)*MAT[:,1])  
    for KX in range(k_num):
        K_PATH = np.append(K_PATH, K_PATH[-3:]-1/k_num*3*np.pi/(2*a)*MAT[:,0]+1/k_num*np.pi/(2*a)*MAT[:,1])  
    K_PATH = K_PATH.reshape(6*k_num+1,3)                  # Array of k-vectors of shape (6*K_num+1, 3) 
    num_kpoints = np.size(K_PATH[:,0])
    for k in range(num_kpoints):
        K_PATH[k,:] = np.dot(M_rotz,K_PATH[k,:]) 
    print("Number of kpoints: " + str(num_kpoints) + " (path)")
    file = open('k_path.txt','w')
    for i in range(num_kpoints):
        for j in range(3):
            file.write("%s " % K_PATH[i][j].real)
        file.write("\n")    
    file.close()
    #print(K_PATH[0], K_PATH[40], K_PATH[80], K_PATH[120], K_PATH[160], K_PATH[200], K_PATH[240])
    return K_PATH
    


def k_irr_BZ():    
    '''
    Calculates the k-vectors of the irreducable/full BZ:qdel 
    '''
    MAT = np.zeros((3,3))                                                     # Matrix of reciprocal basis vectors
    MAT[:,0] = np.array([-1., 1., 1.])*2.*np.pi/a
    MAT[:,1] = np.array([ 1.,-1., 1.])*2.*np.pi/a
    MAT[:,2] = np.array([ 1., 1.,-1.])*2.*np.pi/a
        
    lattice = np.array([[0.0, 0.5, 0.5],                    # basis vectors of fcc lattice
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0]])*a  
                    
    positions = [[0.0, 0.0, 0.0],                           # atomic basis in fractional coordinates
                 [0.5, 0.0, 0.0],
                 [0.0, 0.5, 0.0],
                 [0.0, 0.0, 0.5]]             
       
    numbers= [1,2,3,4]      
        
    cell = (lattice, positions, numbers)
    
    print('spacegroup: ' +str(spglib.get_spacegroup(cell, symprec=1e-5)))
    print(spglib.get_symmetry(cell, symprec=1e-5))    
    
    # caclulatio of irr. BZ vectors + weights
    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0,0,0])
    MAT_help = grid[np.unique(mapping)]/np.array(mesh, dtype=float)
    MAT_irr_BZ = np.zeros((np.size(MAT_help[:,0]),3))       
    for k in range(1,np.size(MAT_help[:,0])):
        MAT_irr_BZ[k,:] = MAT[:,0]*MAT_help[k,0] + MAT[:,1]*MAT_help[k,1] + MAT[:,2]*MAT_help[k,2] # transform from fractional to cartesian coordinates

    print("Number of kpoints: %d (irr BZ)" % len(np.unique(mapping)))
    num_kpoints = np.size(MAT_irr_BZ[:,0])

    weights = (np.unique(mapping,return_counts=True)[1])
    print("Number of kpoints: %d (full BZ, check of weights)" % weights.sum())           
           
    #for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
    #    print("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh))
    #print(grid[np.unique(mapping)]/np.array(mesh, dtype=float))    
    #print(np.unique(mapping,return_counts=True))      
    
    # caclulatio of full BZ vectors (weights = 1) 
    MAT_BZ_full = np.array(grid, dtype=float)
    for k in range(1,np.size(MAT_BZ_full[:,0])):
        MAT_BZ_full[k,:] = MAT[:,0]*MAT_BZ_full[k,0] + MAT[:,1]*MAT_BZ_full[k,1]+ MAT[:,2]*MAT_BZ_full[k,2]
    print("Number of kpoints: %d (full BZ)" % np.size(MAT_BZ_full[:,0]))
    
    file = open('k_BZ_irr.txt','w')
    for i in range(num_kpoints):
        for j in range(3):
            file.write("%s " % MAT_irr_BZ[i][j])
        file.write("\n")    
    file.close()
    
    file = open('k_weights_irr.txt','w')
    for i in range(num_kpoints):
        file.write("%s " % (weights[i]*1.0))
        file.write("\n")    
    file.close()
    
    file = open('k_BZ_full.txt','w')
    for i in range(np.size(MAT_BZ_full[:,0])):
        for j in range(3):
            file.write("%s " % (MAT_BZ_full[i][j]/mesh[0]))
        file.write("\n")    
    file.close()
    
    file = open('k_weights_full.txt','w')
    for i in range(np.size(MAT_BZ_full[:,0])):
        file.write("%s " % 1.0)
        file.write("\n")    
    file.close()
    return MAT_irr_BZ, MAT_BZ_full/(mesh[0]*1.0)


def k_irr_BZ_OCTOPUS(): 
    '''
    irr BZ from Octopus file
    '''
    MAT = np.zeros((3,3))                                   # Matrix of reciprocal basis vectors
    MAT[:,0] = np.array([ -1.,  1.,  1.])*2.*np.pi/a
    MAT[:,1] = np.array([  1., -1.,  1.])*2.*np.pi/a
    MAT[:,2] = np.array([  1.,  1., -1.])*2.*np.pi/a

    file  = open('BZ_OCTOPUS/irr_BZ_OCT.txt','r')
    MAT_BZ = np.loadtxt(file)
    file.close()
    
    file  = open('BZ_OCTOPUS/irr_weights_OCT.txt','r')
    weights = np.loadtxt(file)
    file.close()
    
    file  = open('BZ_OCT_XYZ.txt','w')
    for k in range(np.size(MAT_BZ[:,0])):
        MAT_BZ[k,:] = MAT[:,0]*MAT_BZ[k,0] + MAT[:,1]*MAT_BZ[k,1] + MAT[:,2]*MAT_BZ[k,2]
    for i in range(np.size(MAT_BZ[:,0])):
        for j in range(3):
            file.write("%s " % MAT_BZ[i][j])
        file.write("\n")    
    file.close()
    
    file = open('k_weights_XYZ.txt','w')
    for i in range(np.size(MAT_BZ[:,0])):
        fac = 1./weights[0]
        file.write("%s " % (weights[i]*fac))
        file.write("\n")    
    file.close()
    
    print("Number of kpoints: %d (irr BZ OCTOPUS)" % np.size(MAT_BZ[:,0]))
    return MAT_BZ

K_PATH = k_path()
MAT_irr_BZ, MAT_BZ_full = k_irr_BZ()
#MAT_irr_BZ_OCT = k_irr_BZ_OCTOPUS()

############################################################################### PLOT

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d, Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(K_PATH[:,0], K_PATH[:,1], K_PATH[:,2], c="b", marker="x")
ax.scatter(MAT_irr_BZ[:,0], MAT_irr_BZ[:,1], MAT_irr_BZ[:,2], c="r", marker="x")
ax.scatter(MAT_BZ_full[:,0], MAT_BZ_full[:,1], MAT_BZ_full[:,2], c="k", marker=".")
#ax.scatter(MAT_irr_BZ_OCT[:,0], MAT_irr_BZ_OCT[:,1], MAT_irr_BZ_OCT[:,2], c="g", marker="x")
plt.show()