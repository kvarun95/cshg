import numpy as np

# basic phyical constants
SPEEDOFLIGHT = 299.792458 # um/s
WAVELENGTH = 0.5 # um
NA = 0.65

# length scales
L = 166.4 # um
LZ = 10. # um

def create_grid(N, NZ):

    dL = L/N 
    dL = L/N
    dLZ = LZ/NZ

    # spatial frequency scales
    FMAX = 1./dL # um**-1
    dF = 1./L # um**-1

    # spatial grid
    X = np.arange(N) * dL 
    Z = np.arange(NZ) * dLZ 
    gridx = np.stack(np.meshgrid(X,X), axis=2)

    # spatial frequency grid
    F = np.fft.fftfreq(N, dL)
    F = np.fft.fftshift(F)
    gridf = np.stack(np.meshgrid(F,F), axis=2)

    return {'dL':dL, 'dLZ':dLZ, 'dF':dF, 'FMAX':FMAX, 'Z':Z, 'X': X, 'gridx': gridx, 'F':F, 'gridf':gridf}
    
    