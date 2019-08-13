// Author : Varun Kelkar

// Compile as : 
// g++ -o cfdct3d cfdct3d.cpp libfdct3d.a -fPIC -L/home/varun/fftw-2.1.5/fftw/.libs -lfftw

#include "fdct3d.hpp"
#include "fdct3dinline.hpp"
#include "cfdct3d.hpp"
#include <cstdlib>
#include <ctime>

/*
```
void call_fdct3d(double* xre_io, int n_xre_io,
                double* xim_io, int n_xim_io,
                double* cre_io, int n_cre_io,
                double* cim_io, int n_cim_io,
                int* nxs_io, int n_nxs_io,
                int* nys_io, int n_nys_io,
                int* nzs_io, int n_nzs_io,
                int* W, int n_W,
                int N1, int N2, int N3, 
                int nbscales, int nbdstz_coarse, 
                int ac, double lamda, char option)
```
Function for implementing three fdct3d functions in one function which is interfaced with python using swig. 
Argument:
_input_ `char option` :
... `'P'` : execute `fdct3d_params()`
... `'F'` : execute `fdct3d_forward()`
... `'B'` : execute `fdct3d_inverse()`

_input_ `int N1`, `int N2`, `int N3` : Dimensions of real-space grid.

_input, output_ `int nbscales, int nbdstz_coarse` : Dimensions of the curvelet-space grid. 
... Serves as _output_ for `fdct3d_param` and _input_ for `fdct3d_forward` and `fdct3d_inverse`.

_input, output_ `int ac` : _output_ for `fdct3d_param` and _input_ for `fdct3d_forward` and `fdct3d_inverse`.

_input, output_ `double* xre_in` : pointer to the real part of flattened complex realspace signal. Inactive for `fdct3d_param`,
... Input for `fdct3d_forward`, output for `fdct3d_inverse`.

_input, output_ `double* xim_in` : pointer to the real part of flattened complex realspace signal. Inactive for `fdct3d_param`,
... Input for `fdct3d_forward`, output for `fdct3d_inverse`.

_input,output_ `double* cre_in` : pointer to the real embedded complex curveletspace signal. Inactive for `fdct3d_param`, 
... Output for `fdct3d_forward`, input for `fdct3d_inverse`.

_input,output_ `double* cim_in` : pointer to the real embedded complex curveletspace signal. Inactive for `fdct3d_param`, 
... Output for `fdct3d_forward`, input for `fdct3d_inverse`.
*/
void call_fdct3d(double* xre_io, int n_xre_io,
                double* xim_io, int n_xim_io,
                double* cre_io, int n_cre_io,
                double* cim_io, int n_cim_io,
                int* nxs_io, int n_nxs_io,
                int* nys_io, int n_nys_io,
                int* nzs_io, int n_nzs_io,
                int* W, int n_W,
                int N1, int N2, int N3, 
                int nbscales, int nbdstz_coarse, 
                int ac, double lamda, char option)
{
    srand(time(NULL));

    int S = nbscales;
    

    // get the number of wedges in each scale (is different for each scale)
    if (option=='W') {
        vector< vector<double> > fxs,fys,fzs;
        vector< vector<int> > nxs,nys,nzs;
        fdct3d_param(N1, N2, N3, nbscales, nbdstz_coarse, ac, fxs, fys, fzs, nxs, nys, nzs);
        if (S==n_W && S==nxs.size()) {
            for (int i=0;i<S;i++) {
                W[i] = nxs[i].size();
            }
        }
        else {
            std::cout << "Dimension mismatch : The size of W must match the number of scales S" << std::endl;
        }
    }

    // Get other params : the size of the curvelet coefficients matrix in each wedge of each scale
    if (option=='P') {

        vector< vector<double> > fxs,fys,fzs;
        vector< vector<int> > nxs,nys,nzs;
        fdct3d_param(N1, N2, N3, nbscales, nbdstz_coarse, ac, fxs, fys, fzs, nxs, nys, nzs);

        // conditional statements checking if dimensions are okay
        bool s1=0; bool s2=0;
        if (S!=n_W || S!=nxs.size()) {
            s1 = 1;
            std::cout << "Dimension mismatch : The size of W must match the number of scales S" << std::endl;
        }
        else {
            for (int s=0;s<nxs.size();s++) {
                if (W[s]!=nxs[s].size()) {
                    s2 = s2 + 1;
                    std::cout << "Dimension mismatch : The wedge dimensions contained in W[" << s <<"] should match the wedge dimensions nxs["<< s <<"].size()" << std::endl;
                }
            }
        }

        // if dimensions are okay, populate the integer arrays nxs_io, nys_io, nzs_io with the x,y,z dimensions of curvelet coefficient matices for each wedge for each scale.
        if (!(s1 + s2)) {
            int idx = 0;
            for (int s=0;s<nxs.size();s++) {
                for (int w=0;w<nxs[s].size();w++) {
                    nxs_io[idx] = nxs[s][w];
                    idx++;
                }
            }

            idx = 0;
            std::cout << idx << std::endl;
            for (int s=0;s<nys.size();s++) {
                for (int w=0;w<nys[s].size();w++) {
                    nys_io[idx] = nys[s][w];
                    idx++;
                }
            }

            idx = 0;
            for (int s=0;s<nzs.size();s++) {
                for (int w=0;w<nzs[s].size();w++) {
                    nzs_io[idx] = nzs[s][w];
                    idx++;
                }
            }
        }
    }



    if (option=='F') {
        // Forward curvelet transform. 

        // Populate the x tensor from the input real and imaginary parts
        CpxNumTns x(N1,N2,N3);
        for (int i=0;i<N1;i++) {
            for (int j=0;j<N2;j++) {
                for (int k=0;k<N3;k++) {
                    x(i,j,k) = cpx(xre_io[N2*N3*i+N3*j+k], xim_io[N2*N3*i+N3*j+k]);
                }
            }
        }
        
        vector< vector<CpxNumTns> > c;
        fdct3d_forward(N1, N2, N3, nbscales, nbdstz_coarse, ac, x, c);
        // conditional statements checking if dimensions are okay
        bool s1=0; bool s2=0;
        if (S!=n_W || S!=c.size()) {
            s1 = 1;
            std::cout << "Dimension mismatch : The size of W must match the number of scales S" << std::endl;
        }
        else {
            for (int s=0;s<c.size();s++) {
                if (W[s]!=c[s].size()) {
                    s2 = s2 + 1;
                    std::cout << "Dimension mismatch : The wedge dimensions contained in W[" << s <<"] should match the wedge dimensions nxs["<< s <<"].size()" << std::endl;
                }
            }
        }

        // unpack curvelet coeffs into 1d c++ arrays
        int idx=0;
        int n=0;
        for (int s=0;s<c.size();s++) {
            for (int w=0;w<c[s].size();w++) {
                for (int i=0;i<nxs_io[n];i++) {
                    for (int j=0;j<nys_io[n];j++) {
                        for (int k=0;k<nzs_io[n];k++) {
                            cre_io[idx] = c[s][w](i,j,k).real();
                            cim_io[idx] = c[s][w](i,j,k).imag();
                            idx++;
                        }
                    }
                }
                n++;
            }
        }

    }


    // inverse curvelet transform
    if (option=='B') {
        CpxNumTns x(N1,N2,N3);
        vector< vector<CpxNumTns> > c;        
        // run a forward fdct3d just to get the correct shape for c
        fdct3d_forward(N1, N2, N3, nbscales, nbdstz_coarse, ac, x, c);
        
        // transfer the curvelet coefficients into the curvelet array
        int idx=0;
        int n=0;
        for (int s=0;s<S;s++) {
            for (int w=0;w<W[s];w++) {
                for (int i=0;i<nxs_io[n];i++) {
                    for (int j=0;j<nys_io[n];j++) {
                        for (int k=0;k<nzs_io[n];k++) {
                            c[s][w](i,j,k) = cpx(cre_io[idx], cim_io[idx]);
                            idx++;
                        }
                    }
                }
                n++;
            }
        }

        // empty x 
        clear(x);
        fdct3d_inverse(N1, N2, N3, nbscales, nbdstz_coarse, ac, c, x);
        // unpack real space vector into the 1d arrays
        for (int i=0;i<N1;i++) {
            for (int j=0;j<N2;j++) {
                for (int k=0;k<N3;k++) {
                    xre_io[N2*N3*i+N3*j+k] = x(i,j,k).real();
                    xim_io[N2*N3*i+N3*j+k] = x(i,j,k).imag();
                }
            }
        }

    }


    // Soft thresholding in the curvelet domain
    if (option=='S') {

        // Populate the real space tensor
        CpxNumTns x(N1,N2,N3);
        for (int i=0;i<N1;i++) {
            for (int j=0;j<N2;j++) {
                for (int k=0;k<N3;k++) {
                    x(i,j,k) = cpx(xre_io[N2*N3*i+N3*j+k], xim_io[N2*N3*i+N3*j+k]);
                }
            }
        }
        
        vector< vector<CpxNumTns> > c;
        fdct3d_forward(N1, N2, N3, nbscales, nbdstz_coarse, ac, x, c);
        // conditional statements checking if dimensions are okay
        bool s1=0; bool s2=0;
        if (S!=n_W || S!=c.size()) {
            s1 = 1;
            std::cout << "Dimension mismatch : The size of W must match the number of scales S" << std::endl;
        }
        else {
            for (int s=0;s<c.size();s++) {
                if (W[s]!=c[s].size()) {
                    s2 = s2 + 1;
                    std::cout << "Dimension mismatch : The wedge dimensions contained in W[" << s <<"] should match the wedge dimensions nxs["<< s <<"].size()" << std::endl;
                }
            }
        }

        // soft thresholding on curvelet coefficients
        int idx=0;
        int n=0;
        for (int s=0;s<S;s++) {
            for (int w=0;w<W[s];w++) {
                for (int i=0;i<nxs_io[n];i++) {
                    for (int j=0;j<nys_io[n];j++) {
                        for (int k=0;k<nzs_io[n];k++) {
                            double cre = c[s][w](i,j,k).real();
                            double cim = c[s][w](i,j,k).imag();
                            double are = abs(cre) / sqrt(cre*cre + cim*cim);
                            double aim = abs(cim) / sqrt(cre*cre + cim*cim);
                            if (cre>=lamda*are) cre = cre - lamda*are;
                            if (cre>=-lamda*are && cre<lamda*are) cre = 0.;
                            if (cre<-lamda*are) cre = cre + lamda*are;
                            if (cim>=lamda*aim) cim = cim - lamda*aim;
                            if (cim>=-lamda*aim && cim<lamda*aim) cim = 0.;
                            if (cim<-lamda*aim) cim = cim + lamda*aim;
                            c[s][w](i,j,k) = cpx(cre, cim);
                            idx++;
                        }
                    }
                }
                n++;
            }
        }
        fdct3d_inverse(N1, N2, N3, nbscales, nbdstz_coarse, ac, c, x);

        for (int i=0;i<N1;i++) {
            for (int j=0;j<N2;j++) {
                for (int k=0;k<N3;k++) {
                    xre_io[N2*N3*i+N3*j+k] = x(i,j,k).real();
                    xim_io[N2*N3*i+N3*j+k] = x(i,j,k).imag();
                }
            }
        }

    }

}











