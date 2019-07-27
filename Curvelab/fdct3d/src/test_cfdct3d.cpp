// compile as 
// g++ -o test_cfdct3d test_cfdct3d.cpp cfdct3d.cpp libfdct3d.a -fPIC -L/home/varun/fftw-2.1.5/fftw/.libs -lfftw

#include "cfdct3d.hpp"
#include <numeric> // std::accumulte
#include <cstdlib>
#include <ctime>

void test_param();
void test_forward();

int main() {

    srand(time(NULL));
    // test_param();
    test_forward();

    return 0;
}




void test_param() {

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

}


void test_forward() {

    char option = 'F';
    int ac = 0;
    int nbscales = 2;
    int nbdstz_coarse = 3;
    int n_W = nbscales;
    int* W = new int[n_W];

    int ns = 2;
    int* ns_placeholder = new int[ns];

    int N1 = 256;
    int N2 = 256;
    int N3 = 10;

    int n_xre_io = N1*N2*N3;
    double* xre_io = new double[n_xre_io];
    int n_xim_io = N1*N2*N3;
    double* xim_io = new double[n_xim_io];

    for (int i=0;i<n_xre_io;i++) {
        xre_io[i] = (double)rand() / RAND_MAX;
    }

    int n_cre_io = 2;
    double* cre_io = new double[n_cre_io];
    int n_cim_io = 2;
    double* cim_io = new double[n_cim_io];

    call_fdct3d(xre_io, n_xre_io, xim_io, n_xim_io,
                cre_io, n_cre_io, cim_io, n_cim_io,
                ns_placeholder, ns, ns_placeholder, ns, ns_placeholder, ns,
                W, n_W, N1, N2, N3, nbscales, nbdstz_coarse, ac, 'W');


    for (int i=0;i<n_W;i++) {
        std::cout << "W["<<i<<"] :" << W[i] << std::endl;
    }

    int ns_io = std::accumulate(W, W+n_W, 0);
    int* nxs_io = new int[ns_io];
    int* nys_io = new int[ns_io];
    int* nzs_io = new int[ns_io];

    call_fdct3d(xre_io, n_xre_io, xim_io, n_xim_io,
                cre_io , n_cre_io, cim_io, n_cim_io,
                nxs_io, ns_io, nys_io, ns_io, nzs_io, ns_io,
                W, n_W, N1, N2, N3, nbscales, nbdstz_coarse, ac, 'P');

    int nc = 0;
    for (int i=0;i<ns_io;i++) {
        nc = nc + nxs_io[i]*nys_io[i]*nzs_io[i];
    }

    double* cre = new double[nc];
    double* cim = new double[nc];

    call_fdct3d(xre_io, n_xre_io, xim_io, n_xim_io,
                cre, nc, cim, nc,
                nxs_io, ns_io, nys_io, ns_io, nzs_io, ns_io,
                W, n_W, N1, N2, N3, nbscales, nbdstz_coarse, ac, option);

    for (int i=0;i<nc;i++) {
        std::cout << cre[i] << "," << cim[i] << std::endl;    
    }

}







