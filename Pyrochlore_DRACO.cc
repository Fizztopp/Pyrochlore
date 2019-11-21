/**
 *	TIGHT-BINDING MODEL FOR PYROCHLORE IRIDATES (TBG)
 *  Copyright (C) 2019, Gabriel E. Topp
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 *  02111-1307, USA.
 * 	
 * 	This code is based on a full unit cell tight-binding model for twisted bilayer graphene. For commensurate angles the follwing objects can be calculated
 *  -filling-dependent chenical potential
 *  -equilibrium bands
 *  -equilibrium Berry curvature (2 different methods)
 *  -Floquet spectrum (2 different methods)
 *  -Berry curvature of Floquet states
 * 
 *  Necessary input:
 *  -Unit_Cell.dat: contains atomic positions, and sublattice index
 *  -k_path.dat: list of k-points along high symmetry path
 *  -k_BZ: List of k-points of Brilluoin zone (for reduce also weights are necessary!)  
 */


#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <math.h>
#include <assert.h>
#include <iterator>
#include <sstream>
#include <string>
#include <algorithm>


// PARAMETERS ##########################################################

// Intrinsic parameters
#define NORB      4                            						    // #of orbitals without spin (dimension of matrix is 2*NORB x 2*NORB)
#define U         1.60                                                  // Hubbard repulsion
#define ts        -0.775					      						// hopping
#define alatt     4.0                             						// lattice constant
#define BETA      5.0e1                        						    // inverse temperature
#define filling   4.0                                                   // 4 -> half filling 

// Numerical paramters
#define mu_init   0.2											     	// initial for newton method to find mu
#define dev       1e-15                    					        	// exit deviation for while loop in groundstate() (m)
#define DELTA     1e-5												    // correction prefactor for chemical potential in groundstate()

// Propagation parameters
#define starttime 0.0														
#define endtime   200.0
#define timesteps 2e5

// Quenching parameters
#define U_RAMP    1.45												    // absolute value to which U is ramped
#define T_SHIFT   10.0													// delay after which ramping starts                                 

// Dissipation parameters
#define J0	      0.01 											        // coupling to occupations	1.0e-2
#define BETAB     5.0e1                                                 // bath temperature

// Tr-ARPES
// Probe pulse
#define TIMESTEPS 1e4                                                   // # of timesteps used for G<(t,t') -> timesteps/TIMESTEPS has to be int!
#define T_PROBE 150.0                                                   // delay
#define SIGMA_PROBE 25.0                                                // RMS width
#define OMEGA_PROBE_MIN -0.5                                            // MINIMUM probe frequency
#define OMEGA_PROBE_MAX 0.5                                             // MAXIMUM probe frequency
#define N_OMEGA_PROBE 100                                               // # of probed frequencies
#define K_PROBE_MIN 300                                                 // MINIMUM probe moementum 
#define K_PROBE_MAX 400                                                 // MINIMUM probe moementum
#define weightcutoff 1e-5 											    // cutoff for Gaussian weight defines window where ARPES signal is collected

#define PI 3.14159265359


// CALCULATION OPTIONS #################################################

//#define NO_MPI

#ifndef NO_MPI
    #include <mpi.h>
#endif

//#define NO_DISS


using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

cdouble II(0,1);


// DEFINITION OF FUNCTIONS #############################################

const int DIM = 2*NORB;

//LAPACK (Fortran 90) functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//routine to find eigensystem of Hk
extern "C" {
/** 
 *  Computes the eigenvalues and, optionally, the eigenvectors for a Hermitian matrices H
 */
    void zheev_(char* jobz, char* uplo, int* N, cdouble* H, int* LDA, double* W, cdouble* work, int* lwork, double* rwork, int *info);
}
//'N','V':  Compute eigenvalues only, and eigenvectors
char    jobz = 'V';       
//'U','L':  Upper, Lower triangle of H is stored 
char    uplo = 'U';  
// The order of the matrix H.  DIM >= 0
int     matsize = DIM;    
// The leading dimension of the array H.  lda >= max(1,DIM)
int     lda = DIM;             
// The length of the array work.  lwork  >= max(1,2*DIM-1)
int     lwork = 2*DIM-1;    
// dimension (max(1, 3*DIM-2))
double  rwork[3*DIM-2];  
// dimension (MAX(1,LWORK))
cdouble work[2*DIM-1];  
// Info
int	    info;
            
void diagonalize(cvec &Hk, dvec &evals)
{
/**
 *  Diagonalization of matrix Hk. Stores eigenvalues in real vector evals and eigenvectors in complex vector Hk
 *  -Hk: Complex vector[DIM*DIM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[DIM] to store eigenvalues
 */
    zheev_(&jobz, &uplo, &matsize, &Hk[0], &lda, &evals[0], &work[0], &lwork, &rwork[0], &info);
	assert(!info);
}


//INLINE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inline int fq(int i, int j, int N)
/**
 *  MAT[i,j] = Vec[fq(i,j,N)] with row index i and column index j
 */
{
    return i*N+j;
}


inline int fb(int a, int b, int i, int j)
/**
    MAT[a, b, i, j] = Vec[fq(a,b,i,j)] element; a,b in {0,1,2,3} chooses site; i,j in {0,1} chooses spin coordinate, 
 */
{
	return 8*a + b + 32*i + 4*j;
}


inline double delta(int a, int b)
/**
 *  Delta function
 */
{
	if (a==b)
		return 1.;
	else
		return 0.;
}


inline dvec add(dvec vec1, dvec vec2)
/**
 *  Add vec1 and vec2:
 */
{
	for(int m=0; m<int(vec1.size()); m++)
		vec1[m] = vec1[m] + vec2[m];
	return vec1;
}


inline dvec mult(dvec vec, double a)
/**
 *  Multiplication of vector with scalar:
 */
{
	for(int m=0; m<int(vec.size()); m++)
		vec[m] = vec[m]*a;
	return vec;
}


inline double dot(dvec vec1, dvec vec2)
/**
 *  Scalar product of vectors vec1 and vec2:
 */
{
	return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];
}


inline dvec cross(dvec vec1, dvec vec2)
{
/**
 *  Cross-product between vec1 and vec2:
 */
	dvec vec(3);
	vec[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
	vec[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
	vec[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
	return vec;
}


inline dvec b_i(dvec &MAT_BASIS,  int i)
/**
 *	Basis vectors for alatt = 4:
 * 	-MAT_BASIS: Basis vectors in array of dim[12]
 * 	-i: int in {0,1,2,3}
 */
{
	dvec vec(3);
	for(int m=0; m<3; m++)
		vec[m] = MAT_BASIS[fq(i,m,3)];
	return vec;
}


inline dvec b_ij(dvec &MAT_BASIS,  int i, int j)
/**
 *	Bond vectors:
 * 	-MAT_BASIS: Basis vectors in array of dim[12]
 * 	-i,j: int in {0,1,2,3}
 */
{
	dvec vec(3);
	for(int m=0; m<3; m++)
		vec[m] = MAT_BASIS[fq(j,m,3)] - MAT_BASIS[fq(i,m,3)];
	return vec;
}


inline dvec a_ij(dvec &MAT_BASIS,  int i, int j)
/**
 *	Vectors points from middle of unit cell to half of bond vector b_ij:
 * 	-MAT_BASIS: Basis vectors in array of dim[12]
 * 	-i, j: int in {0,1,2,3}
 */
{
	dvec vec(3);
	for(int m=0; m<3; m++)
		vec[m] = 0.5*(MAT_BASIS[fq(j,m,3)] + MAT_BASIS[fq(i,m,3)]) - 0.5;
	return vec;
}


inline dvec d_ij(dvec &MAT_BASIS,  int i, int j)
/**
 *	Orbit vectors:
 *  -MAT_BASIS: Basis vectors in array of dim[12]
 *  -i, j: int in {0,1,2,3}
 */
{
	return cross(mult(a_ij(MAT_BASIS, i, j), 2.0), b_ij(MAT_BASIS, i, j));
}


inline cvec sigma_dot_vec(dvec vprod)
/**
 *  Calculates product of Pauli matrix-vector {s0, s1, s2} with 3d vector {v1, v2, v3}:
 */
{
	cvec vec(4);
	vec[0] = vprod[2];
	vec[1] = vprod[0] - II*vprod[1];
	vec[2] = vprod[0] + II*vprod[1];
	vec[3] = -vprod[2];
	return vec; 
}


template <class Vec>
inline void print(Vec vec)
/**
 *	print out vector
 */
{
	for(int i=0; i<vec.size(); i++)
		{
	    	cout << vec[i] << " ";
	    }	
	cout << endl;
}


template <class Vec>
inline double get_deviation(Vec &M1, Vec &M2)
/**
 *	calculates deviation of two different desnisties M1 und M2
 */
{
    double deviations = 0.;
	for(int i=0; i<M1.size(); i++)
	{
		deviations += abs(abs(M1[i])-abs(M2[i]));  	
    }
    return deviations;
}


inline double fermi(double energy, double mu)
{
/**
 *	Fermi distribution:
 *	-energy: energy eigenvalue
 *	-mu: chemical potential
 */
    return 1./(exp((energy-mu)*BETA) + 1.);
}


inline double fermi_bath(double energy, double mu)
{
/**
 *	Fermi distribution:
 *	-energy: energy eigenvalue
 *	-mu: chemical potential
 */
    return 1./(exp((energy-mu)*BETAB) + 1.);
}


inline double U_ramp(double time)
{
	if(time <= T_SHIFT)
		return U;
	else if(time > T_SHIFT)	
		return U_RAMP;	
}


// VOID FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

template <class Vec>
void times(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product of quadratic matrices: $C = A \cdot B$
 */
{
    int dim = sqrt(A.size());
	Vec TEMP(dim*dim);
    // Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
	    for(int j=0; j<dim; j++) {
		    TEMP[fq(j,i,dim)] = B[fq(i,j,dim)];
		   }
    }
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += A[fq(i,k,dim)]*TEMP[fq(j,k,dim)]; 
			}
		}
	}	
}


template <class Vec>
void times_dn(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of first factor: $C = A^\dagger \cdot B$
 */
{
	int dim = sqrt(A.size());
	Vec TEMP1(dim*dim);
	Vec TEMP2(dim*dim);
	// Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
		for(int j=0; j<dim; j++) {
			TEMP1[fq(j,i,dim)] = A[fq(i,j,dim)];
			TEMP2[fq(j,i,dim)] = B[fq(i,j,dim)];
		}
	}		
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += conj(TEMP1[fq(i,k,dim)])*TEMP2[fq(j,k,dim)];
			}
		}
	}		
}


template <class Vec>
void times_nd(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of second factor: $C = A \cdot B^\dagger$
 */
{
	int dim = sqrt(A.size());	
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
					C[fq(i,j,dim)] += A[fq(i,k,dim)]*conj(B[fq(j,k,dim)]);
			}
		}
	}	
}


void ReadIn(vector<dvec> &MAT, const string& filename)
{
/**
 *	Read in real valued matrix
 */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		dvec row((istream_iterator<double>(is)),
		istream_iterator<double>());
		MAT.push_back(row);
	}
	in.close();
}


void ReadInC(vector<cvec> &MAT, const string& filename)
{
/**
 *	Read in complex valued matrix
 */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		cvec row((istream_iterator<cdouble>(is)),
		istream_iterator<cdouble>());
		MAT.push_back(row);
	}
	in.close();
}


void Dens_MF(int num_kpoints_BZ, double num_kpoints_BZ_full, vector<cvec> &RHO_0, cvec &DENS, vector<dvec> &kweights, int &numprocs, int &myrank)
/**
 *	Calculates the average spin density for every site
 *	-num_kpoints_BZ: # of k-points in irr. BZ
 * 	-num_kpoints_BZ_full: # of k-points of full BZ
 *	-RHO_0: Vector of complex vectors[64] for intital desity matrix
 * 	-DENS: array which contains pseudo spins [dim,orbital]
 * 	-kweights: weights of k-points
 * 	-numprocs: # of processes 
 * 	-myrank: rank of process
 */
{
	for(int i=0; i<16; i++)
		DENS[i] = 0.;
		
	for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
	{		
		for(int a=0; a<NORB; a++)
		{		
			DENS[fq(a, 0, 4)] += 1./num_kpoints_BZ_full*(RHO_0[k][fb(a, a, 0, 0)])*(kweights[k][0]);       // Spin up density 
			DENS[fq(a, 1, 4)] += 1./num_kpoints_BZ_full*(RHO_0[k][fb(a, a, 0, 1)])*(kweights[k][0]);	   // Spin up/down density 	
			DENS[fq(a, 2, 4)] += 1./num_kpoints_BZ_full*(RHO_0[k][fb(a, a, 1, 0)])*(kweights[k][0]);	   // Spin down/up density
			DENS[fq(a, 3, 4)] += 1./num_kpoints_BZ_full*(RHO_0[k][fb(a, a, 1, 1)])*(kweights[k][0]);       // Spin down density
		}		
	}
}


void M_MF(int num_kpoints_BZ, double num_kpoints_BZ_full, vector<cvec> &RHO_0, dvec &M, vector<dvec> &kweights, int &numprocs, int &myrank)
/**
 *	Calculation of mean field pseudspin vectors
 *  -num_kpoints_BZ: # of k-points in BZ_IRR
 *  -num_kpoints_BZ_full: # of k-points of full BZ
 *	-RHO_0: Vector of complex vectors[64] for intital desity matrix
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -kweights: Weights of k-points
 *  -numprocs: # of processes 
 *  -myrank: Rank of process
 */
{
	for(int i=0; i<12; i++)
		M[i] = 0.;
		
	for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
	{		
		for(int a=0; a<NORB; a++)
		{		
			M[fq(0,a,4)] += real(1./(num_kpoints_BZ_full*2.)*(RHO_0[k][fb(a, a, 0, 1)]+RHO_0[k][fb(a, a, 1, 0)]))*(kweights[k][0]);                // mx
			M[fq(1,a,4)] += real(1./(num_kpoints_BZ_full*2.)*(-II)*(RHO_0[k][fb(a, a, 1, 0)]-RHO_0[k][fb(a, a, 0, 1)]))*(kweights[k][0]);	   	   // my
			M[fq(2,a,4)] += real(1./(num_kpoints_BZ_full*2.)*(RHO_0[k][fb(a, a, 0, 0)]-RHO_0[k][fb(a, a, 1, 1)]))*(kweights[k][0]);	               // mz
		}		
	}
}


void set_Hk(dvec &kvec, dvec &M, cvec &Hk, dvec &MAT_BASIS, double time)
/**
 *  k-dependent Hamiltonian:
 *  -kvec: Real vector of the 3d irreducible Brilluoin zone
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -MAT_BASIS: Basis vectors in array of dim[12]
 *  -time: Real time coordinate
 */
{
	// microscopic motivated hopping parameters
	double tp  = -2.*ts/3.;
	double tss = ts*0.08;
	double tps = tp*0.08;
	double t1  = 0.53 + 0.27*ts;
	double t2  = 0.12 + 0.17*ts;
	double t1s = 233./2916.*tss - 407./2187.*tps;
	double t2s = 1./1458.*tss + 220./2187.*tps;
	double t3s = 25./1458.*tss + 460./2187.*tps;
	
	for(int i=0; i<64; i++)    											
	{
		Hk[i] = 0.;
	}	
		
	for(int a=0; a<NORB; a++)
	{	
		Hk[fb(a,a,0,0)] += -U_ramp(time)*(M[fq(2,a,4)]  - (M[fq(0,a,4)]*M[fq(0,a,4)] + M[fq(1,a,4)]*M[fq(1,a,4)] + M[fq(2,a,4)]*M[fq(2,a,4)]));           // spin up-up
		Hk[fb(a,a,0,1)] += -U_ramp(time)*(M[fq(0,a,4)]-II*M[fq(1,a,4)]);																				  // spin up-down
		Hk[fb(a,a,1,0)] += -U_ramp(time)*(M[fq(0,a,4)]+II*M[fq(1,a,4)]);																				  // spin down-up	    
		Hk[fb(a,a,1,1)] += -U_ramp(time)*(-M[fq(2,a,4)] - (M[fq(0,a,4)]*M[fq(0,a,4)] + M[fq(1,a,4)]*M[fq(1,a,4)] + M[fq(2,a,4)]*M[fq(2,a,4)]));  	      // spin down-down
	
		for(int b=0; b<NORB; b++)
		{
			Hk[fb(a,b,0,0)] += 2.*(t1+t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[0])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));
			Hk[fb(a,b,0,1)] += 2.*(                   t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[1])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));
			Hk[fb(a,b,1,0)] += 2.*(                   t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[2])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));
			Hk[fb(a,b,1,1)] += 2.*(t1+t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[3])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));

			for(int c=0; c<NORB; c++)
			{
				Hk[fb(a,b,0,0)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((t1s*(1.-delta(a,b)) + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[0])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
				Hk[fb(a,b,0,1)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((                    + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[1])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
				Hk[fb(a,b,1,0)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((                    + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[2])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
				Hk[fb(a,b,1,1)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((t1s*(1.-delta(a,b)) + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[3])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
			}
		}	
	}	
}


void set_Hk_kin(dvec &kvec, dvec &M, cvec &Hk, dvec &MAT_BASIS)
/**
 *  k-dependent kinetic Hamiltonian:
 *  -kvec: Real vector of the 3d irreducible Brilluoin zone
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -MAT_BASIS: Basis vectors in array of dim[12]
 */
{
	// microscopic motivated hopping parameters
	double tp  = -2.*ts/3.;
	double tss = ts*0.08;
	double tps = tp*0.08;
	double t1  = 0.53 + 0.27*ts;
	double t2  = 0.12 + 0.17*ts;
	double t1s = 233./2916.*tss - 407./2187.*tps;
	double t2s = 1./1458.*tss + 220./2187.*tps;
	double t3s = 25./1458.*tss + 460./2187.*tps;
	
	for(int i=0; i<64; i++)    											
	{
		Hk[i] = 0.;
	}	
		
	for(int a=0; a<NORB; a++)
	{	
		for(int b=0; b<NORB; b++)
		{
			Hk[fb(a,b,0,0)] += 2.*(t1+t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[0])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));
			Hk[fb(a,b,0,1)] += 2.*(   t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[1])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));
			Hk[fb(a,b,1,0)] += 2.*(   t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[2])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));
			Hk[fb(a,b,1,1)] += 2.*(t1+t2*II*sigma_dot_vec(d_ij(MAT_BASIS, a, b))[3])*cos(dot(kvec, b_ij(MAT_BASIS, a, b)));

			for(int c=0; c<NORB; c++)
			{
				Hk[fb(a,b,0,0)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((t1s*(1.-delta(a,b)) + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[0])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
				Hk[fb(a,b,0,1)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((                    + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[1])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
				Hk[fb(a,b,1,0)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((                    + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[2])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
				Hk[fb(a,b,1,1)] += 2.*(1.-delta(a,c))*(1.-delta(b,c))*((t1s*(1.-delta(a,b)) + II*sigma_dot_vec(add(mult(cross(b_ij(MAT_BASIS, a, c),b_ij(MAT_BASIS, c, b)),t2s), mult(cross(d_ij(MAT_BASIS, a, c), d_ij(MAT_BASIS, c, b)),t3s)))[3])*cos(dot(kvec,add(b_ij(MAT_BASIS, c, a),b_ij(MAT_BASIS, c, b)))));
			}
		}	
	}
}


void set_Hk_pot(dvec &M, cvec &Hk, double time)
/**
 *  k-independent inteaction part of Hamiltonian:
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -time: real time coordinate for calling time-dependent U by function U_ramp(time)
 */
{
	// Microscopic motivated hopping parameters
	double tp  = -2.*ts/3.;
	double tss = ts*0.08;
	double tps = tp*0.08;
	double t1  = 0.53 + 0.27*ts;
	double t2  = 0.12 + 0.17*ts;
	double t1s = 233./2916.*tss - 407./2187.*tps;
	double t2s = 1./1458.*tss + 220./2187.*tps;
	double t3s = 25./1458.*tss + 460./2187.*tps;
	
	for(int i=0; i<64; i++)    											
	{
		Hk[i] = 0.;
	}	
		
	for(int a=0; a<NORB; a++)
	{	
		Hk[fb(a,a,0,0)] += -U_ramp(time)*(M[fq(2,a,4)]  - (M[fq(0,a,4)]*M[fq(0,a,4)] + M[fq(1,a,4)]*M[fq(1,a,4)] + M[fq(2,a,4)]*M[fq(2,a,4)]));           // spin up-up
		Hk[fb(a,a,0,1)] += -U_ramp(time)*(M[fq(0,a,4)]-II*M[fq(1,a,4)]);																				  // spin up-down
		Hk[fb(a,a,1,0)] += -U_ramp(time)*(M[fq(0,a,4)]+II*M[fq(1,a,4)]);																				  // spin down-up	    
		Hk[fb(a,a,1,1)] += -U_ramp(time)*(-M[fq(2,a,4)] - (M[fq(0,a,4)]*M[fq(0,a,4)] + M[fq(1,a,4)]*M[fq(1,a,4)] + M[fq(2,a,4)]*M[fq(2,a,4)]));  	      // spin down-down
	}
}


void groundstate(vector<dvec> &kweights, vector<cvec> &RHO_0, dvec &M, cvec &Hk, dvec &evals, vector<dvec> &BZ_IRR, dvec &MAT_BASIS, double &mu, int &numprocs, int &myrank)
/**
 *	Computes the ground state pseudospin vectors and density matrix in self-consistent loop
 *  -kweights: Real vector containing weights of k-points (in case reduced cell is used unequal 1)
 *	-RHO_0: Vector of complex vectors[64] for intital desity matrix
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -evals: Real vector[8] of eigenvalues
 *  -BZ_IRR: k-points of reduced reciprocal cell
 *  -MAT_BASIS: Vector of real vectors containing basis vectors
 *  -mu: Chemical potential
 *  -numprocs: Total number of processes (MPI)
 *  -myrank: Rank of process (MPI)
 */
{
    // # of k-vectors from sampling of irreducible BZ
	int num_kpoints_BZ = BZ_IRR.size();                                                         
	
	// K-oints of full Brillouin zone
	double num_kpoints_BZ_full = 0.;
	for(int k=0; k<kweights.size(); k++)
		num_kpoints_BZ_full += kweights[k][0];						
	
	int count = 0;
	double m = 0.0;
	double m_old;	    
	double deviation = 1.0;
	double N_tot, E_tot; 
	cvec TEMP(64,0.);
	
	// Set chemical potential to guessed inittial value
	mu = mu_init;
		
	while(deviation > dev)
	{
		count++;
		m_old = m;	
		N_tot = 0.;
		E_tot = 0.;
		
		// Calculate particle number per k-point
		for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
		{	
		    // Set Hamiltonian matrix
			set_Hk(BZ_IRR[k], M, Hk, MAT_BASIS, 0.0);
			// Diagonalize Hamiltonian matrix
			diagonalize(Hk, evals); 
			// Set density matrix in eigenenergy basis and calculate total particel number    
			for(int i=0; i<8; i++){
				for(int j=0; j<8; j++)
				{
					RHO_0[k][fq(i,j,8)] = fermi(evals[i], mu)*delta(i,j); 
					N_tot += fermi(evals[i], mu)*delta(i,j)*kweights[k][0];
					E_tot += fermi(evals[i], mu)*delta(i,j)*kweights[k][0]*evals[i]; 	  
				}	   
			}  
			times(RHO_0[k], Hk, TEMP);                                              
	        times_dn(Hk, TEMP, RHO_0[k]);                      
		}
            	
		// Calculation of mean-field pseudospin vectors 
		M_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, M, kweights, numprocs, myrank);
		
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &N_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &E_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &M[0], 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
		// Adjust chemical potential
		mu += -DELTA*(N_tot-num_kpoints_BZ_full*filling);	

        // Calculation of magnetic order parameter
		m = 0.;
		for(int a=0; a<4; a++)
		{
			m += 0.25*sqrt(M[fq(0,a,4)]*M[fq(0,a,4)] + M[fq(1,a,4)]*M[fq(1,a,4)] + M[fq(2,a,4)]*M[fq(2,a,4)]);
		}
		
		// Deviation from former loop
		deviation = abs(m-m_old);
		
		if(myrank==0){
			cout << "loop #" << count << ": deviation = " << deviation << endl;
			cout << "mx: " << M[fq(0,0,4)] << " " << M[fq(0,1,4)] << " " << M[fq(0,2,4)] << " " << M[fq(0,3,4)] << endl;
			cout << "my: " << M[fq(1,0,4)] << " " << M[fq(1,1,4)] << " " << M[fq(1,2,4)] << " " << M[fq(1,3,4)] << endl;
			cout << "mz: " << M[fq(2,0,4)] << " " << M[fq(2,1,4)] << " " << M[fq(2,2,4)] << " " << M[fq(2,3,4)] << endl;
			cout << "magnetization per site m = " << m << endl;	
			cout << "total particle number N = " << N_tot << endl;
			cout << "average particle number per k = " << N_tot/num_kpoints_BZ_full << endl;
			cout << "total energy = " << E_tot << endl;
			cout << "Average energy per particle = " << E_tot/(4.*num_kpoints_BZ_full) << endl;
			cout << "chemical potential mu = " << mu << endl;
			cout << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
		}	 	 
	}
	ofstream myfile;
	if(myrank==0)
	{
		myfile.open("mu.txt");
		if (myfile.is_open())
		{
			myfile << mu << endl;
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
	
		myfile.open("m0.txt");
		if (myfile.is_open())
		{
			myfile << m << endl;
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
		
		myfile.open("E_TOT_0.txt");
		if (myfile.is_open())
		{
			myfile << E_tot << endl;
			myfile.close();
		}
		else cout << "Unable to open file" << endl;		
	}	
}


void set_dRHOdt(cvec &TEMP1, cvec &TEMP2, cvec &RHO_t_tk, cvec &dRHO_dt, cvec &Hk, dvec &evals, double &mu, double time)
/**
 *  Calculation of the time-derivative of the density matrix
 *  -TEMP1, TEMP2: Complex helper matrix 
 *  -RHO_t_tk:Complex vector[64] of k- and time-dependent density matrix
 *  -dRHO_dt: Complex vector[64] of temporal change of density matrix
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -evals: Real vector[8] of eigenvalues
 *  -mu: chemical potential
 *  -time: double real time coordinate
 */
{	
	// COHERENT PART
	times(Hk, RHO_t_tk, TEMP1);										
	times(RHO_t_tk, Hk, TEMP2);
	for(int i=0; i<64; i++)
	{
		dRHO_dt[i] = -II*(TEMP1[i]-TEMP2[i]);	
	}
	// DISSIPATION PART	
#ifndef NO_DISS
    diagonalize(Hk, evals);

	for(int i=0; i<64; i++)
		TEMP1[i] = RHO_t_tk[i];											// copy RHO(t) to TEMP1
	
    // Transform Rho: orbital basis -> band basis
	times_nd(TEMP1, Hk, TEMP2);
	times(Hk, TEMP2, TEMP1);										    // S @ RHO_ORB @ S^(-1) = RHO_BAND 
	
	// Calculate dRho
	for(int a=0; a<8; a++)
	{
		for(int b=0; b<8; b++)
		{
			TEMP2[fq(a,b,8)] = -2.*J0*(TEMP1[fq(a,b,8)]-fermi_bath(evals[a], mu))*delta(a,b) - 2.*J0*TEMP1[fq(a,b,8)]*(1.-delta(a,b));
		}
	}		
	
	// Transform dRho: band basis -> orbital basis
	times(TEMP2, Hk, TEMP1);                                            // S^(-1) @ RHO_BAND @ S = RHO_ORBITAL
	times_dn(Hk, TEMP1, TEMP2);			
	
	for(int i=0; i<64; i++)
	{
		dRHO_dt[i] += TEMP2[i];
	}	
#endif 		
}


void AB2_propatation(dvec &mu_t, vector<dvec> &E_TOT, dvec &evals, vector<dvec> &kweights, vector<cvec> &RHO_0, vector<cvec> &dRHO_dt0, vector<cvec> &dRHO_dt1, dvec &M, cvec &Hk, vector<dvec> &BZ_IRR, dvec &MAT_BASIS, vector<cvec*> RHO_t, vector<dvec*> M_t, double &mu, int &numprocs, int &myrank)
/**
 *	Two-step Adams-Bashforth linear multistep propagator:
 *	-mu_t: Real vector to store t.-d. chemical potential
 *	-E_TOT: Vector[timesteps] of real vectors[3] to store t.-d. energies
 *  -evals: Real vector[8] of eigenvalues
 *	-kweights: Real vector containing weights of k-points (in case reduced cell is used unequal 1)*
 *	-RHO_0: Vector of complex vectors[64] for intital desity matrix
 *	-dRHO_dt0, dRHO_dt1: Vector of complex vectors[64] to store change of density matrix
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -Hk: Complex vector[64] to store Hamiltonian
 *	-BZ_IRR: k-points of reduced reciprocal cell
 *	-MAT_BASIS: Real vector[12] of basis vectors
 *	-RHO_t_tk: Vector[3] of complex vector pointers[64] to store density matrix at 3 subsequent time steps
 *	-mu: chemical potential
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI) 
 */
{
	// # of k-vectors from sampling of irreducible BZ
	int num_kpoints_BZ = BZ_IRR.size();                                 
	// k-oints of full Brillouin zone   	
		int num_kpoints_BZ_full = 0;
	for(int k=0; k<kweights.size(); k++)
		num_kpoints_BZ_full += kweights[k][0];							
	
	// Temporary variables
	double m;;
	cvec E(3);                                                          // vector to store enegies of each time step
	cvec DENSt(16,0.);																		
	cvec TEMP1(64,0.);												    // Helper arrays for set_dRhodt()
	cvec TEMP2(64,0.);
	cvec TEMP3(64,0.);			
	double n_tot;
	dvec N_TOT(timesteps);	
	cvec *temp0, *temp1, *temp2;
	
	// Stepsize
	double h = (endtime-starttime)/timesteps;	

	// Initial magnetization
	for(int i=0; i<12; i++)
	{
		(*M_t[0])[i] = M[i];
	}
			
	for(int k=0; k<num_kpoints_BZ; k++)
	{
		for(int i=0; i<64; i++)
			(*RHO_t[fq(0, k, num_kpoints_BZ)])[i] = RHO_0[k][i];
	}
	
	mu_t[0] = mu;
	N_TOT[0] = filling;
	
	for(int i=0; i<3; i++){
			E[i] = 0.0;
	}
	for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
	{
		set_Hk(BZ_IRR[k], M, Hk, MAT_BASIS, 0.0);
		times(RHO_0[k], Hk, TEMP1);
		for(int i=0; i<8; i++)
		{
			E[0] += TEMP1[fq(i,i,8)]*kweights[k][0];	
		}
		// Calculation of potential energy	
		set_Hk_pot(M, Hk, 0.0);
		times(RHO_0[k], Hk, TEMP1);
		for(int i=0; i<8; i++)
		{
			E[1] += TEMP1[fq(i,i,8)]*kweights[k][0];	
		}
		// Calculation of kinetic energy	
		set_Hk_kin(BZ_IRR[k], M, Hk, MAT_BASIS);
		times(RHO_0[k], Hk, TEMP1);
		for(int i=0; i<8; i++)
		{
			E[2] += TEMP1[fq(i,i,8)]*kweights[k][0];	
		}
	}
	Dens_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, DENSt, kweights, numprocs, myrank);
	
#ifndef 	NO_MPI		
	MPI_Allreduce(MPI_IN_PLACE, &DENSt[0], 16, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &E[0], 3, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif		
	for(int i=0; i<3; i++)
	{
		E[i]=E[i]/(4.0*num_kpoints_BZ_full);
		E_TOT[0][i] = real(E[i]);
	}
	n_tot = 0.0;
	for(int a=0; a<4; a++)
	{
		n_tot += real(DENSt[fq(a,0,4)]+DENSt[fq(a,3,4)]);
	}
	N_TOT[0] = n_tot;
	
	double time;
	// Propagation	
	for(int t=0; t<timesteps-1; t++)
	{   
		for(int i=0; i<3; i++){
			E[i] = 0.0;
		}	
		n_tot = 0.0;
		if(t==0)
		{       
			for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
			{ 
				set_Hk(BZ_IRR[k], M, Hk, MAT_BASIS, h*double(t));
				set_dRHOdt(TEMP1, TEMP2, RHO_t[fq(0,k,num_kpoints_BZ)][0], dRHO_dt0[k], Hk, evals, mu_t[0], h*double(t));
				// 1st Euler step: y_{n+1} = y_n + hf(t_{n},y_{n})	
				for(int i=0; i<64; i++)
				{
					(*RHO_t[fq(1,k,num_kpoints_BZ)])[i] = (*RHO_t[fq(0,k,num_kpoints_BZ)])[i] + h*dRHO_dt0[k][i]; 		
					RHO_0[k][i] = (*RHO_t[fq(1,k,num_kpoints_BZ)])[i];   
				}	
			}			
			// Calculations of mean-field magnetization order
			M_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, M, kweights, numprocs, myrank);
			// Calculations of mean-field operators
			Dens_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, DENSt, kweights, numprocs, myrank);
			
#ifndef 	NO_MPI		
			MPI_Allreduce(MPI_IN_PLACE, &M[0], 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif		
			for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
			{ 	
				// Calculation of total energy	
				set_Hk(BZ_IRR[k], M, Hk, MAT_BASIS, h*double(t));
				times(RHO_0[k], Hk, TEMP1);
				for(int i=0; i<8; i++)
				{
					E[0] += TEMP1[fq(i,i,8)]*kweights[k][0];	
				}
				// Calculation of potential energy	
				set_Hk_pot(M, Hk, h*double(t));
				times(RHO_0[k], Hk, TEMP1);
				for(int i=0; i<8; i++)
				{
					E[1] += TEMP1[fq(i,i,8)]*kweights[k][0];	
				}
				// Calculation of kinetic energy	
				set_Hk_kin(BZ_IRR[k], M, Hk, MAT_BASIS);
				times(RHO_0[k], Hk, TEMP1);
				for(int i=0; i<8; i++)
				{
					E[2] += TEMP1[fq(i,i,8)]*kweights[k][0];	
				}
			}
#ifndef 	NO_MPI		
			MPI_Allreduce(MPI_IN_PLACE, &DENSt[0], 16, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &E[0], 3, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif							
			if(myrank==0) cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<< endl;
			if(myrank==0) cout << "U(t): " << U_ramp(h*double(t)) << endl;  
			for(int i=0; i<12; i++)
			{
				(*M_t[1])[i] = M[i];
			}
			for(int i=0; i<3; i++)
			{
				E[i]=E[i]/(4.0*num_kpoints_BZ_full);
				E_TOT[t+1][i] = real(E[i]);
			}
			for(int a=0; a<4; a++)
			{
				n_tot += real(DENSt[fq(a,0,4)]+DENSt[fq(a,3,4)]);
			}
			N_TOT[t+1] = n_tot;
			// Adjust chemical potential in order to preserve particle number
			mu_t[t+1] = mu_t[t]*(N_TOT[0]/N_TOT[t+1]);	 
			if(myrank==0) cout << "E_tot: " << E[0] << " E_pot: " << E[1] << " E_kin: " << E[2] << "------------------------------------------" << endl;
		}
		// Two step Adams–Bashforth method
		else
		{	// 2-step Adams predictor	
			for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
			{
				set_Hk(BZ_IRR[k], M_t[t-1][0], Hk, MAT_BASIS, h*double(t-1));					
				set_dRHOdt(TEMP1, TEMP2, RHO_t[fq(0,k,num_kpoints_BZ)][0], dRHO_dt0[k], Hk, evals, mu_t[t-1], h*double(t-1));
				set_Hk(BZ_IRR[k], M_t[t][0], Hk, MAT_BASIS, h*double(t));					
				set_dRHOdt(TEMP1, TEMP2, RHO_t[fq(1,k,num_kpoints_BZ)][0], dRHO_dt1[k], Hk, evals, mu_t[t], h*double(t));
				// P_{n+1} = y_{n} + 3/2*h*f(t_{n},y_{n}) - 0.5*h*f(t_{n-1},y_{n-1})
				for(int i=0; i<64; i++)
				{
					(*RHO_t[fq(2,k,num_kpoints_BZ)])[i] = (*RHO_t[fq(1,k,num_kpoints_BZ)])[i] + h*(3./2.*dRHO_dt1[k][i] - 0.5*dRHO_dt0[k][i]); 		
					RHO_0[k][i] = (*RHO_t[fq(2,k,num_kpoints_BZ)])[i]; 		
				}
			}			
			
			// Calculations of mean-field operators				
			M_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, M, kweights, numprocs, myrank); 

#ifndef 	NO_MPI		
			MPI_Allreduce(MPI_IN_PLACE, &M[0], 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif				
			// 2-step Moulton corrector: 
			for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
			{
				set_Hk(BZ_IRR[k], M, Hk, MAT_BASIS, h*double(t+1));				
				set_dRHOdt(TEMP1, TEMP2, RHO_t[fq(2,k,num_kpoints_BZ)][0], dRHO_dt0[k], Hk, evals, mu_t[t], h*double(t+1));
				 // y_{n+1} = y_{n} + 1/2*h*(f(t_{n+1},P_{n+1}) + f(t_{n},y_{n}))
				for(int i=0; i<64; i++)
				{
					(*RHO_t[fq(2,k,num_kpoints_BZ)])[i] = (*RHO_t[fq(1,k,num_kpoints_BZ)])[i] + 0.5*h*(dRHO_dt0[k][i] + dRHO_dt1[k][i]); 		  
					RHO_0[k][i] = (*RHO_t[fq(2,k,num_kpoints_BZ)])[i]; 	
				}
			}	
			M_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, M, kweights, numprocs, myrank);
			Dens_MF(num_kpoints_BZ, num_kpoints_BZ_full, RHO_0, DENSt, kweights, numprocs, myrank);	
	
#ifndef 	NO_MPI		
			MPI_Allreduce(MPI_IN_PLACE, &M[0], 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
			for(int k=myrank; k<num_kpoints_BZ; k+=numprocs)
			{	
				// Calculation of total energy	
				set_Hk(BZ_IRR[k], M, Hk, MAT_BASIS, h*double(t));	
				times(RHO_0[k], Hk, TEMP1);
				for(int i=0; i<8; i++)
				{
					E[0] += TEMP1[fq(i,i,8)]*kweights[k][0];	
				}
				// Calculation of potential energy	
				set_Hk_pot(M, Hk, h*double(t));
				times(RHO_0[k], Hk, TEMP1);
				for(int i=0; i<8; i++)
				{
					E[1] += TEMP1[fq(i,i,8)]*kweights[k][0];	
				}
				// Calculation of kinetic energy	
				set_Hk_kin(BZ_IRR[k], M, Hk, MAT_BASIS);
				times(RHO_0[k], Hk, TEMP1);
				for(int i=0; i<8; i++)
				{
					E[2] += TEMP1[fq(i,i,8)]*kweights[k][0];	
				}
				
				// Cyclic exchange of pointers
				temp0 = RHO_t[fq(0,k,num_kpoints_BZ)];
				temp1 = RHO_t[fq(1,k,num_kpoints_BZ)];
				temp2 = RHO_t[fq(2,k,num_kpoints_BZ)];
				
				RHO_t[fq(0,k,num_kpoints_BZ)] = temp1;
				RHO_t[fq(1,k,num_kpoints_BZ)] = temp2;
				RHO_t[fq(2,k,num_kpoints_BZ)] = temp0;
			}	
#ifndef 	NO_MPI		
			MPI_Allreduce(MPI_IN_PLACE, &DENSt[0], 16, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &E[0], 3, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif							
			if(myrank==0)cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<< endl;
			
			for(int i=0; i<12; i++)
			{
				(*M_t[t+1])[i] = M[i];
			}	
			for(int i=0; i<3; i++)
			{
				E[i]=E[i]/(4.0*num_kpoints_BZ_full);
				E_TOT[t+1][i] = real(E[i]);
			}	
			m = 0;
			for(int a=0; a<4; a++)
			{
				m += 0.25*sqrt(M[fq(0,a,4)]*M[fq(0,a,4)] + M[fq(1,a,4)]*M[fq(1,a,4)] + M[fq(2,a,4)]*M[fq(2,a,4)]);
			}	
			n_tot = 0.0;
			for(int a=0; a<4; a++)
			{
				n_tot += real(DENSt[fq(a,0,4)]+DENSt[fq(a,3,4)]);
			}
			N_TOT[t+1] = n_tot;
			if(myrank==0) 
			{	
				cout << "time step: " << t <<  endl;
				cout << "propagated time: " << h*double(t) <<  endl;
				cout << "U(t): " << U_ramp(h*double(t)) << endl;
				cout << "pseudo-spin orientation: ------------------------------------------------------------------------------------------" << endl;
				cout << "mx: " << M[fq(0,0,4)] << " " << M[fq(0,1,4)] << " " << M[fq(0,2,4)] << " " << M[fq(0,3,4)] << endl;
				cout << "my: " << M[fq(1,0,4)] << " " << M[fq(1,1,4)] << " " << M[fq(1,2,4)] << " " << M[fq(1,3,4)] << endl;
				cout << "mz: " << M[fq(2,0,4)] << " " << M[fq(2,1,4)] << " " << M[fq(2,2,4)] << " " << M[fq(2,3,4)] << endl;
				cout << "Mean-field spin-density: ------------------------------------------------------------------------------------------" << endl;
				cout << "a=1: " << DENSt[fq(0,0,4)] << " " << DENSt[fq(0,1,4)] << " " << DENSt[fq(0,2,4)] << " " << DENSt[fq(0,3,4)] << endl;
				cout << "a=2: " << DENSt[fq(1,0,4)] << " " << DENSt[fq(1,1,4)] << " " << DENSt[fq(1,2,4)] << " " << DENSt[fq(1,3,4)] << endl;
				cout << "a=3: " << DENSt[fq(2,0,4)] << " " << DENSt[fq(2,1,4)] << " " << DENSt[fq(2,2,4)] << " " << DENSt[fq(2,3,4)] << endl;
				cout << "a=4: " << DENSt[fq(3,0,4)] << " " << DENSt[fq(3,1,4)] << " " << DENSt[fq(3,2,4)] << " " << DENSt[fq(3,3,4)] << endl;		
				cout << "total magnetization per site m = " << m << endl;
				cout << "td chemical potential mu(t) = " << mu_t[t] << endl;
				cout << "average particle # per k-point n_k = " << n_tot << endl;
				cout << "E_tot: " << E[0] << " E_pot: " << E[1] << " E_kin: " << E[2] << endl;
				cout << "----------------------------------------------------------------------------------------------------------------------------------------" << endl;
			}
						
		}
		// Adjust chemical potential in order to preserve particle number
		mu_t[t+1] = mu_t[t]*(N_TOT[0]/N_TOT[t+1]);	                    
	}
	// Write td mean field parameters to file
	ofstream myfile;
	if(myrank==0)
	{
		myfile.open("E_t.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps; t++)
			{
				for(int i=0; i<3; i++)
					myfile << E_TOT[t][i] << " ";
				myfile  << endl;	
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	

		myfile.open("U_t.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps; t++)
			{
				myfile <<  U_ramp(h*double(t)) << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	

		myfile.open("M_t.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps-1; t++)
			{
				for(int i=0; i<12; i++)
				{
					myfile << (*M_t[t])[i] << " ";
				}
			myfile  << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;
			
		myfile.open("m_t_MF.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps-1; t++)
			{   
				m = 0;
				for(int a=0; a<4; a++)
				{
					m += 0.25*sqrt((*M_t[t])[fq(0,a,4)]*(*M_t[t])[fq(0,a,4)] + (*M_t[t])[fq(1,a,4)]*(*M_t[t])[fq(1,a,4)] + (*M_t[t])[fq(2,a,4)]*(*M_t[t])[fq(2,a,4)]);
				}		
				myfile  << m << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;

		myfile.open("n_t.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps-1; t++)
			{   
				myfile  << N_TOT[t] << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;

		myfile.open("mu_t.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<timesteps-1; t++)
			{   
				myfile  << mu_t[t] << endl;
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
}


void Hk_bands(vector<dvec> BANDS, dvec &M,  cvec &Hk, dvec &evals, vector<dvec> &K_PATH, dvec &MAT_BASIS, double time, const string& filename)
/**
 *	Calculate bands of Hk(k) for high symmetry path and store them in "bands.txt":
 *	-BANDS: Vector of real vectors[3] to store k-points of path
 *  -M: Real[12] vector contains the 4 mean-field 3d pseudospin vectors
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -evals: Real vector[8] of eigenvalues
 *	-K_PATH: k-points of reduced reciprocal cell
 *	-MAT_BASIS: Real vector[12] of basis vectors
 *	-time: real time coordinate
 *	-filename: String to define file
 */
{
	int num_kpoints_path = K_PATH.size();

	for(int k=0; k<num_kpoints_path; k++)
	{
		set_Hk(K_PATH[k], M, Hk, MAT_BASIS, time);
		diagonalize(Hk, evals);
		for(int m=0; m<8; m++)
			BANDS[k][m] = evals[m];
	}
	ofstream myfile (filename);
	if (myfile.is_open())
	{
		for(int k=0; k<num_kpoints_path; k++)
		{
			for(int m=0; m<8; m++)
			{
				myfile << BANDS[k][m] << " " ;
			}
		myfile  << endl;
		}
	myfile.close();
	}
    else cout << "Unable to open file" << endl;
}


void set_Rhok(cvec &RHOk, cvec &Hk, dvec &evals, double &mu)
{	
/**
 *	Calculates the thermal density operator RHOk(k) in orbital basis for a given Hamiltonian Hk
 * 	-RHOk: density operator in band basis dim[64]
 *  -Hk: Complex vector[64] to store Hamiltonian
 *	-evals: Real vector[8] of eigenvalues  
 *	-mu: chemical potential
 */	
	cvec temp(64,0.);
	cvec nk0(64,0.);
	
	diagonalize(Hk, evals);                                             // Hk -> S (eigenvectors as columns)

	for(int i=0; i<8; i++)
		nk0[fq(i,i,8)] = fermi(evals[i], mu);         	

	times(nk0, Hk, temp);                                               // S @ H @ S^(-1) = H_D --> nk = S^(-1) @ nk_D @ S
	times_dn(Hk, temp, RHOk);
}


void PROP_PATH(int k, dvec &evals, cvec &Hk, vector<dvec> &K_PATH, dvec &MAT_BASIS, vector<dvec> M_t, vector<cvec*> RHO_PATH_t, dvec &mu_t)
/**
 *	Caclulates the time-dependent density matrices for k in K_PATH:
 *	-k: choses k vector
 *  -evals: Real vector[8] of eigenvalues
 *	-K_PATH: vector of high-symmetry path vectors
 *	-MAT_BASIS: Basis vectors in array of dimension [4][3]
 *	-M_t: td Magnetization vectors
 *	-RHO_PATH_t: Matrix to store density
 *	-mu_t: t.-d. chemical potential 
 */
{
	 // # of k-vectors from sampling of irreducible BZ
	int num_kpoints_PATH = K_PATH.size();                              
																	
	cvec TEMP1(64,0.);												    
	cvec TEMP2(64,0.);
	cvec dRHO_dt0(64);
	cvec dRHO_dt1(64);
	
	double h = (endtime-starttime)/timesteps;

	// Set initial density matrix
	set_Hk(K_PATH[k], M_t[0], Hk, MAT_BASIS, 0.0);
	set_Rhok(TEMP1, Hk, evals, mu_t[0]);
	for(int i=0; i<64; i++)
	{
		(*RHO_PATH_t[0])[i] = TEMP1[i];
	}	
	
	// Propagation	
	for(int t=0; t<timesteps-1; t++)
	{   
		//if(myrank==0) cout << "time step (PATH): " << t << " ";
		// 1st Euler step: // y_{n+1} = y_{n} +  h*f(t_{n},y_{n}) 
		if(t==0)
		{      
			set_Hk(K_PATH[k], M_t[0], Hk, MAT_BASIS, h*double(t));
			set_dRHOdt(TEMP1, TEMP2, RHO_PATH_t[t][0], dRHO_dt0, Hk, evals, mu_t[0], 0.0);
			for(int i=0; i<64; i++)
			{
				(*RHO_PATH_t[t+1])[i] = (*RHO_PATH_t[t])[i] + h*dRHO_dt0[i]; 		                        
			}
		}
		// Two step Adams–Bashforth method
		else
		{	// 2-step Adams predictor: P_{n+1} = y_{n} + 3/2*h*f(t_{n},y_{n}) - 0.5*h*f(t_{n-1},y_{n-1})		

			set_Hk(K_PATH[k], M_t[t-1], Hk, MAT_BASIS, h*double(t-1));					
			set_dRHOdt(TEMP1, TEMP2, RHO_PATH_t[t-1][0], dRHO_dt0, Hk, evals, mu_t[t-1], h*double(t-1));
			set_Hk(K_PATH[k], M_t[t], Hk, MAT_BASIS, h*double(t));					
			set_dRHOdt(TEMP1, TEMP2, RHO_PATH_t[t][0], dRHO_dt1, Hk, evals, mu_t[t], h*double(t));
				
			for(int i=0; i<64; i++)
			{
				(*RHO_PATH_t[t+1])[i] = (*RHO_PATH_t[t])[i] + h*(3./2.*dRHO_dt1[i] - 0.5*dRHO_dt0[i]); 	
			}			
			// 2-step Moulton corrector: y_{n+1} = y_{n} + 1/2*h*(f(t_{n+1},P_{n+1}) + f(t_{n},y_{n}))

			set_Hk(K_PATH[k], M_t[t+1], Hk, MAT_BASIS, h*double(t+1));				
			set_dRHOdt(TEMP1, TEMP2, RHO_PATH_t[t+1][0], dRHO_dt0, Hk, evals, mu_t[t+1], h*double(t+1));
				
			for(int i=0; i<64; i++)
			{
				(*RHO_PATH_t[t+1])[i] = (*RHO_PATH_t[t])[i] + 0.5*h*(dRHO_dt0[i] + dRHO_dt1[i]); 		
			}
		}	
	}				
}


inline double gauss(double time, double delay, double sigma)
/**
 *	Gauss distribution function
 *	-time: real time coordinate
 * 	-delay: shift of mean value
 * 	-sigma: sigma value
 */
{
	return 1./(sigma*sqrt(2.*PI))*exp(-0.5*pow((time-delay)/sigma,2.));
}


void Tr_Gless(int k, dvec &mu_t, cvec &Hk, dvec &evals, vector<cvec> &UMATRIX, dvec &MAT_BASIS, double &mu, vector<dvec> &K_PATH, vector<cvec> &G_HELP, cvec &G_Tr, vector<cvec*> RHO_PATH_t, int &myrank)
/**
 *	Calculates the trace of the lesser Grrens funtion Tr{G<(k,t,t')}
 *	-k: Integer picks k-point from K_PATH
 *	-mu_t: Real vector to store t.-d. chemical potential
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -evals: Real vector[8] of eigenvalues
 * 	-UMATRIX: Vector of complex matrices to store unitary mid-point Euler propagators
 * 	-MAT_BASIS: Basis vectors in array of dimension [4][3]
 *  -mu: Chemical potential of initial state
 *  -K_PATH: vector of high-symmetry path vectors
 *  -G_HELP: Vector of complex vectors[64] needed in computation process
 * 	-G_Tr: Complex vector[TIMESTEPS*TIMESTEPS] to store trace of Glesser function
 * 	-RHO_PATH_t: Vector of complex vector pointers containing propagated density matrices of K_PATH
 *  -myrank: Rank of process (MPI) 
 */
{
	dvec M0(12,0.);
	cvec TEMP1(64); 
	cvec TEMP2(64); 
	vector<dvec> M_t(timesteps, dvec(12));
	
	// Load time-dependent magnetic order from prior calculation
	ifstream in("M_t.txt");
	if (!in) 
	{
		cout << "Cannot open file.\n";
		return;
	}
	for(int t=0; t<timesteps-1; t++)
	{
		for(int i=0; i<12; i++)
		{
			in >> M_t[t][i];			
		}	
	}
	in.close();
	in.clear();
	
	// Load time-dependent chemical potential
	in.open("mu_t.txt");
	if (!in) 
	{
		cout << "Cannot open file.\n";
		return;
	}
	for(int t=0; t<timesteps-1; t++)
	{
		in >> mu_t[t];			
	}
	in.close();
	in.clear();
	
	// Propagate 
	PROP_PATH(k, evals, Hk, K_PATH, MAT_BASIS, M_t, RHO_PATH_t, mu_t);
	
	// Stepsize for reduced number timesteps
	double h = (endtime-starttime)/TIMESTEPS;
	int time_fac = timesteps/TIMESTEPS;
	
	//	Set unitary mid-point Euler propagators in k-dependent orbital basis:
    for(int t=0; t<TIMESTEPS-1; t++)
	{
		set_Hk(K_PATH[k], M_t[t*time_fac], TEMP1, MAT_BASIS, h*double(t));           
		set_Hk(K_PATH[k], M_t[(t+1)*time_fac], TEMP2, MAT_BASIS, h*double(t+1));	
		for(int i=0; i<64; i++)
			Hk[i] = 0.5*(TEMP1[i]+TEMP2[i]);
		diagonalize(Hk, evals); 
		for(int i=0; i<8; i++)
		{
			for(int j=0; j<8; j++)
			{
				TEMP1[fq(i,j,8)] = exp(+II*evals[i]*h)*delta(i,j);
			}	
		}
		// Back-transformation to original k-orbital basis
		times(TEMP1, Hk, TEMP2);                                        
		times_dn(Hk, TEMP2, UMATRIX[t]);	
	}	

	// Clear memeory 	
	for(int tt=0; tt<TIMESTEPS*TIMESTEPS; tt++)	
		G_Tr[tt] = 0.0;
	
	// Set G<(t,tp)
	for(int td=0; td<TIMESTEPS-1; td++)
	{
		// Set diagonal (t==t") value of Greens function: G<(t,t') = i*rho(t)
		for(int i=0; i<64; i++)
		{
			G_HELP[td][i] = II*(*RHO_PATH_t[td*time_fac])[i];	
		}
		// Calculate trace
		G_Tr[fq(td, td, TIMESTEPS)] = 0.0;		
			for(int i=0; i<8; i++)
				G_Tr[fq(td, td, TIMESTEPS)] += G_HELP[td][fq(i,i,8)];		
		
		// Propagation in t direction
		for(int t=0; t<TIMESTEPS-td-1; t++)
		{
			times(G_HELP[td+t], UMATRIX[td+t], G_HELP[td+t+1]);
			G_Tr[fq(td+t+1, td, TIMESTEPS)] = 0.0;		
			for(int i=0; i<8; i++)
				G_Tr[fq(td+t+1, td, TIMESTEPS)] += G_HELP[td+t+1][fq(i,i,8)];	
		}
		// Propagation in t' direction
		for(int tp=0; tp<TIMESTEPS-td-1; tp++)
		{
			times_dn(UMATRIX[td+tp], G_HELP[td+tp], G_HELP[td+tp+1]);
			G_Tr[fq(td, td+tp+1, TIMESTEPS)] = 0.0;		
			for(int i=0; i<8; i++)
				G_Tr[fq(td, td+tp+1, TIMESTEPS)] += G_HELP[td+tp+1][fq(i,i,8)];		
		}	
	}		
	if(myrank==0)
	{
		ofstream myfile ("G_tr.txt");
		if (myfile.is_open())
		{
			for(int t=0; t<TIMESTEPS; t++)
			{
				for(int tp=0; tp<TIMESTEPS; tp++)
					myfile <<  G_Tr[fq(t, tp, TIMESTEPS)] << " ";
				myfile  << endl;	
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
	}
}	 



double Iphoto(double omega, cvec &G_Tr) 
/**
 *	Calculate Photo current from Tr_Gless()
 * 	-omega: probing frequency
 *  -G_Tr: Complex vector[TIMESTEPS x TIMESTEPS] to store trace of Glesser function
 **/
{
	double h = (endtime-starttime)/TIMESTEPS;
	double Iph;
	dvec TEMP(TIMESTEPS);
	
	for(int tp=0; tp<TIMESTEPS; tp++)
	{
		TEMP[tp] = 0.0;
		for(int t=0; t<TIMESTEPS-1; t++)
		{
			if(gauss(double(t)*h, T_PROBE, SIGMA_PROBE)*gauss(double(tp)*h, T_PROBE, SIGMA_PROBE)*(2.*PI*pow(SIGMA_PROBE, 2))<weightcutoff)
				continue;
			TEMP[tp] += 0.5*h*imag( gauss(double(t)*h, T_PROBE, SIGMA_PROBE)*gauss(double(tp)*h, T_PROBE, SIGMA_PROBE)*exp(-II*omega*double(t-tp)*h)*G_Tr[fq(t, tp, TIMESTEPS)]
			                       +gauss(double(t+1)*h, T_PROBE, SIGMA_PROBE)*gauss(double(tp)*h, T_PROBE, SIGMA_PROBE)*exp(-II*omega*double(t+1-tp)*h)*G_Tr[fq(t+1, tp, TIMESTEPS)]);
		}	  
	}	
	Iph = 0.0;	
	for(int tp=0; tp<TIMESTEPS-1; tp++)
	{
		Iph += 0.5*h*(TEMP[tp]+TEMP[tp+1]);
	}			
	return Iph;
}	


void EDC(cvec &G_Tr, dvec &IPHOTO, int &myrank)
/**
 * 	Calculation of Energy Distribution Curve
 *  -G_Tr: Complex vector[TIMESTEPS*TIMESTEPS] to store trace of Glesser function
 *  -IPHOTO: Real vector[N_OMEGA_PROBE] to store frequency-dependent photocurrent
 *  -myrank: Rank of process (MPI) 
 */
{
	dvec OMEGA(N_OMEGA_PROBE);
	for(int w=0; w<N_OMEGA_PROBE; w++)
	{
		OMEGA[w] =  OMEGA_PROBE_MIN + (OMEGA_PROBE_MAX - OMEGA_PROBE_MIN)/N_OMEGA_PROBE*double(w);
	    IPHOTO[w] = Iphoto(OMEGA[w], G_Tr); 
		if(myrank==0) cout << "w#" << w << " = " << OMEGA[w] << endl; 
	}	
}	


void EDC_MPI(cvec &G_Tr, dvec &IPHOTO, int &numprocs, int &myrank)
/**
 *	Calculate of Energy Distribution Curve (MPI parallel)
 *  -G_Tr: Complex vector[TIMESTEPS*TIMESTEPS] to store trace of Glesser function
 *  -IPHOTO: Real vector[N_OMEGA_PROBE] to store frequency-dependent photocurrent
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
{
	dvec OMEGA(N_OMEGA_PROBE);
	for(int w=myrank; w<N_OMEGA_PROBE; w+=numprocs)
	{
		OMEGA[w] =  OMEGA_PROBE_MIN + (OMEGA_PROBE_MAX - OMEGA_PROBE_MIN)/N_OMEGA_PROBE*double(w);
	    IPHOTO[w] = Iphoto(OMEGA[w], G_Tr); 
		if(myrank==0) cout << "w#" << w << " = " << OMEGA[w] << endl; 
	}	
	MPI_Allreduce(MPI_IN_PLACE, &IPHOTO[0], N_OMEGA_PROBE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	if(myrank==0)
	{
		ofstream myfile ("EDC.txt");
		if (myfile.is_open())
		{
			for(int w=0; w<N_OMEGA_PROBE; w++)
			{   
				myfile  << IPHOTO[w] << endl;
			}	
			myfile.close();
		}
    else cout << "Unable to open file" << endl;
	}
}	


void TrARPES(cvec &Hk, dvec &mu_t, dvec &evals, vector<cvec> &UMATRIX, dvec &MAT_BASIS, double &mu, vector<dvec> &K_PATH, vector<cvec> &G_HELP, cvec &G_Tr, dvec &IPHOTO, vector<cvec*> RHO_PATH_t, dvec &ARPES, int &numprocs, int &myrank)
/**
 *	Calculation of time-resolved frequency- and momentum-dependent phtotcurrent
 *  -Hk: Complex vector[64] to store Hamiltonian
 *  -mu_t: Real vector to store t.-d. chemical potential
 *  -evals: Real vector[8] of eigenvalues
 * 	-UMATRIX: Vector of complex matrices to store unitary mid-point Euler propagators
 * 	-MAT_BASIS: Basis vectors in array of dimension [4][3]
 *  -mu: Chemical potential of initial state
 *  -K_PATH: vector of high-symmetry path vectors
 *  -G_HELP: Vector of complex vectors[64] needed in computation process
 * 	-G_Tr: Complex vector[TIMESTEPS*TIMESTEPS] to store trace of Glesser function
 *  -IPHOTO: Real vector[N_OMEGA_PROBE] to store frequency-dependent photocurrent
 * 	-RHO_PATH_t: Vector of complex vector pointers containing propagated density matrices of K_PATH
 *  -ARPES: Real vector[(K_PROBE_MAX-K_PROBE_MI) x N_OMEGA_PROBE] to store photocurrent
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
{
	for(int k=K_PROBE_MIN+myrank; k<K_PROBE_MAX; k+=numprocs)
	{
		if(myrank==0) cout << "k = " << k << endl; 
		Tr_Gless(k, mu_t, Hk, evals, UMATRIX, MAT_BASIS, mu, K_PATH, G_HELP, G_Tr, RHO_PATH_t, myrank);
		EDC(G_Tr, IPHOTO, myrank);	
		for(int w=0; w<N_OMEGA_PROBE; w++)
		{		
			ARPES[fq(w, k-K_PROBE_MIN, K_PROBE_MAX-K_PROBE_MIN)] = IPHOTO[w];
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &ARPES[0], N_OMEGA_PROBE*(K_PROBE_MAX-K_PROBE_MIN), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	// Print to file	
	if(myrank==0)
	{
		//print(ARPES);
		ofstream myfile ("ARPES.txt");
		if (myfile.is_open())
		{
			for(int w=0; w<N_OMEGA_PROBE; w++)
			{
				for(int k=0; k<K_PROBE_MAX-K_PROBE_MIN; k++)
				{
					myfile << ARPES[fq(N_OMEGA_PROBE-w-1,k,K_PROBE_MAX-K_PROBE_MIN)] << " " ;
				}
			myfile  << endl;
			}	
			myfile.close();
		}
    else cout << "Unable to open file" << endl;
	}
}	

// main() function #####################################################

int main(int argc, char * argv[])
{
    //************** MPI INIT ***************************
  	int numprocs=1, myrank=0, namelen;
    
#ifndef NO_MPI
  	char processor_name[MPI_MAX_PROCESSOR_NAME];
  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	MPI_Get_processor_name(processor_name, &namelen);
    
	cout << "Process " << myrank << " on " << processor_name << " out of " << numprocs << " says hello." << endl;
	MPI_Barrier(MPI_COMM_WORLD);
    
#endif
	if(myrank==0) cout << "\n\tProgram running on " << numprocs << " processors." << endl;

   
	// DECLARATION AND INTITALIZATION
  
	// matrix of basis-vectors of fcc lattice
	dvec MAT_BASIS = {0.,0.,0.,0.,1.,1.,1.,0.,1.,1.,1.,0.};
	
	
	//vector of high-symmetry path vectors
	vector<dvec> K_PATH;
	ReadIn(K_PATH, "k_path.txt");
	if(myrank==0) cout << "high-symmetry path --> " << K_PATH.size() << " points" << endl;
	int num_kpoints_PATH = K_PATH.size();
	
	//vector of weights
	vector<dvec> kweights;
	ReadIn(kweights, "k_weights_irr.txt");
			
	//vector of BZ vectors
	vector<dvec> BZ_IRR;
	ReadIn(BZ_IRR, "k_BZ_irr.txt");
	if(myrank==0) cout << "irreducible BZ --> " << BZ_IRR.size() << " points" << endl;
	int num_kpoints_BZ = BZ_IRR.size();

	// vector for eigenvalues
	dvec evals(8);

	// vector for Hamiltonian Hk
	cvec Hk(64);
	
	// bands
	vector<dvec> BANDS(K_PATH.size(),dvec(8));
 	
	// allocation of matrices RHO[k]
	vector<cvec> RHO_0(num_kpoints_BZ, cvec(64,0.0));                                       		
	
	// chemical potential
	double mu;															//intial mu
	dvec mu_t(timesteps);												//time-dependent mu (needed to keep particle number constant for bath coupling)
	
	// Averaged spin densities
	cvec DENS(16);
	
    // initial Magnetization M
    dvec M(12);
	dvec MAT_ni = {-1.,-1.,+1.,+1.,-1.,+1.,-1.,+1.,-1.,+1.,+1.,-1.};    // vector of all-out configuration  {-1.,-1.,+1.,+1.,-1.,+1.,-1.,+1.,-1.,+1.,+1.,-1.}
	for(int i=0; i<12; i++)
	{
		M[i] = 1./sqrt(3.)*MAT_ni[i];                        
	}
	
	// allocation of matrix dRHO_dt[k,t]
	vector<cvec> dRHO_dt0(num_kpoints_BZ, cvec(64));  
	vector<cvec> dRHO_dt1(num_kpoints_BZ, cvec(64));                                     		
	
	// dynamic allocation of matrix RHO[k,t]
	vector<cvec*> RHO_t(num_kpoints_BZ*3);                              //<-- 2 step A-B need 
	for(int kt=0; kt<num_kpoints_BZ*3; kt++)
		RHO_t[kt] = new cvec(64);	
				
	// dynamic allocation for td-magnetization
	vector<dvec*> M_t(timesteps);                                       
	for(int t=0; t<timesteps; t++)
		M_t[t] = new dvec(12);	
	
	// Total energy
	vector<dvec> E_TOT(timesteps, dvec(3));						
	
	// Allocations for trARPES calculations
	vector<cvec> UMATRIX(TIMESTEPS,cvec(64));                           // matrix where all instantanouse eigenenergy values are stored
	vector<cvec> G_HELP(TIMESTEPS, cvec(64));                           // matrix with timesteps*8x8 matrices to store G<(t,t_const')
               
	cvec G_Tr(TIMESTEPS*TIMESTEPS);	                                    // matrix of dim timesteps*timesteps to store Tr{G<(k,t,tp)}                       
	dvec IPHOTO(N_OMEGA_PROBE);
	dvec ARPES(N_OMEGA_PROBE*(K_PROBE_MAX-K_PROBE_MIN));
	
	vector<cvec*> RHO_PATH_t(timesteps);               //<-- 2 step A-B need 
	for(int t=0; t<timesteps; t++)
		RHO_PATH_t[t] = new cvec(64);	

		
	// CALCULATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	const clock_t begin_time = clock();
	
	if(myrank==0){cout << "Start caluclation of chemical potential and intial density operator" << endl;}
	groundstate(kweights, RHO_0, M, Hk, evals,  BZ_IRR, MAT_BASIS, mu, numprocs, myrank);
	
	if(myrank==0) 
	{
		print(M);
		if(myrank==0){cout << "Start caluclation of equilibrium bands" << endl;}
		Hk_bands(BANDS, M,  Hk, evals, K_PATH, MAT_BASIS, starttime, "bands0.txt");
	}
	
    if(myrank==0){cout << "Store working parameters" << endl;}
	if(myrank==0)
	{
		double stepsize = (endtime-starttime)/timesteps;
		ofstream myfile ("parameters.txt");
		if (myfile.is_open())
		{
			myfile << mu << " " << ts << " " << U << " " << U_RAMP << " " << 0.0 << " " << BETA << " " << starttime  << " " << endtime  << " " << timesteps << " " << stepsize << " " << TIMESTEPS << " " << T_PROBE << " " << SIGMA_PROBE << " " << OMEGA_PROBE_MIN << " " << OMEGA_PROBE_MAX << " " << N_OMEGA_PROBE << " " << K_PROBE_MIN << " " << K_PROBE_MAX;
			myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}

	if(myrank==0){cout << "Start propagation of senisty operator" << endl;}	
	AB2_propatation(mu_t, E_TOT, evals, kweights, RHO_0, dRHO_dt0, dRHO_dt1, M, Hk, BZ_IRR, MAT_BASIS, RHO_t, M_t, mu, numprocs, myrank);
	
	if(myrank==0){cout << "Start propagation of senisty operator" << endl;}		
	TrARPES(Hk, mu_t, evals, UMATRIX, MAT_BASIS, mu, K_PATH, G_HELP, G_Tr, IPHOTO, RHO_PATH_t, ARPES, numprocs, myrank); 
	
	if(myrank==0) cout << "Total calculations time: " << float(clock() - begin_time)/CLOCKS_PER_SEC << " seconds" << endl;
		
#ifndef NO_MPI
	MPI_Finalize();
#endif	
	
	// free memory	
	for(int kt=0; kt<num_kpoints_BZ*3; kt++)
	{                            
		delete RHO_t[kt];
	}	
	for(int t=0; t<timesteps; t++)
	{                            
		delete M_t[t];
	}
	for(int t=0; t<timesteps; t++)
	{                            
		//delete M_t[t];
		delete RHO_PATH_t[t];
	}

}
