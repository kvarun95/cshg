// Author : Varun Kelkar

// Compile as : 
// g++ -o cfdct3d cfdct3d.cpp libfdct3d.a -fPIC -L/home/varun/fftw-2.1.5/fftw/.libs -lfftw

#include "fdct3d.hpp"
#include "fdct3dinline.hpp"
#include "cfdct3d.hpp"


int main() {

    char option = 'P';
    int ac = 0;
    int nbscales = 3;
    int nbdstz_coarse = 3;
    int n_W = 3;
    int* W = new int[n_W];
    W[0] = 1; W[1] = 54; W[2] = 1; 

    int n_nxs_io = 56;
    int n_nys_io = 56;
    int n_nzs_io = 56;
    int* nxs_io = new int[n_nxs_io];
    int* nys_io = new int[n_nys_io];
    int* nzs_io = new int[n_nzs_io];

    int N1 = 256;
    int N2 = 256;
    int N3 = 10;
    int n_xre_io = N1*N2*N3;
    int n_xim_io = N1*N2*N3;
    double* xre_io = new double[n_xre_io];
    double* xim_io = new double[n_xim_io];
    
    int n_cre_io = 2;
    int n_cim_io = 2;
    double* cre_io = new double[n_cre_io];
    double* cim_io = new double[n_cim_io];


    call_fdct3d(xre_io, n_xre_io, 
                xim_io, n_xim_io, 
                cre_io, n_cre_io, 
                cim_io, n_cim_io, 
                nxs_io, n_nxs_io,
                nys_io, n_nys_io,
                nzs_io, n_nzs_io,
                W, n_W, N1, N2, N3, nbscales, nbdstz_coarse, ac, option);

    std::cout << "nbscales :" << nbscales << std::endl;
    for (int i=0;i<n_W;i++) {
        std::cout << "W[" << i << "] :" << W[i] << std::endl;
    }
    for (int i=0;i<56;i++) {
        std::cout << "n[" << i << "] : (" << nxs_io[i] << "," << nys_io[i] << "," << nzs_io[i] << ")" << std::endl;
    }

    return 0;
}


/*
```
void fdct3dpipe(char option, int N1, int N2, int N3, int nbscales, int nbdstz_coarse, int ac,
                double* x, double* c)
```
Function for implementing three fdct3d functions in one function which is interfaced with python using swig. 
Argument:
_input_ `char option` :
... `'P'` : execute `fdct3d_params()`
... `'F'` : execute `fdct3d_forward()`
... `'I'` : execute `fdct3d_inverse()`

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
                int ac, char option)
{

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



    // if (option=='F') {

    //     CpxNumTns x(N1,N2,N3);
    //     for (int i=0;i<N1;i++) {
    //         for (int j=0;j<N2;j++) {
    //             for (int k=0;k<N3;k++) {
    //                 x(i,j,k) = cpx(xre_io[N2*N3*i+N3*j+k], xim_io[N2*N3*i+N3*j+k]);
    //             }
    //         }
    //     }




    // }

}











