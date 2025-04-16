from . import backend as bd
import torch

def Epsilon_fft(dN,eps_grid,G, batch_eps):
    '''dN = 1/Nx/Ny
    For now, assume epsilon is diagonal; if epsilon has xz,yz component, just simply add them to off-diagonal eps2
    
    eps_grid is  (1) for isotropic, a numpy 2d array in the format of (Nx,Ny),
                 (2) for anisotropic, a list of numpy 2d array [(Nx,Ny),(Nx,Ny),(Nx,Ny)]
    '''
    if batch_eps:
        if eps_grid.shape[-1] == 3 and eps_grid[0][0].ndim == 2:
            raise NotImplementedError
        elif eps_grid[0][0].ndim == 1:
            eps_fft = get_conv(dN,eps_grid,G, batch_eps)
            epsinv = bd.inv(eps_fft)
        
            tmp1 = torch.cat((eps_fft,bd.zeros_like(eps_fft)), dim=1)
            tmp2 = torch.cat((bd.zeros_like(eps_fft),eps_fft), dim=1)
            eps2 = torch.cat((tmp1,tmp2), dim=2)
        else:
            raise ValueError("Wrong eps_grid type")
    else:
        if len(eps_grid) == 3 and eps_grid[0].ndim == 2:
            epsx_fft = get_conv(dN,eps_grid[0],G, batch_eps)
            epsy_fft = get_conv(dN,eps_grid[1],G, batch_eps)
            epsz_fft = get_conv(dN,eps_grid[2],G, batch_eps)
            epsinv = bd.inv(epsz_fft)
        
            tmp1 = bd.vstack((epsx_fft,bd.zeros_like(epsx_fft)))
            tmp2 = bd.vstack((bd.zeros_like(epsx_fft),epsy_fft))
            eps2 = bd.hstack((tmp1,tmp2))
            
        elif eps_grid[0].ndim == 1:
            eps_fft = get_conv(dN,eps_grid,G, batch_eps)
            epsinv = bd.inv(eps_fft)
        
            tmp1 = bd.vstack((eps_fft,bd.zeros_like(eps_fft)))
            tmp2 = bd.vstack((bd.zeros_like(eps_fft),eps_fft))
            eps2 = bd.hstack((tmp1,tmp2))
        else:
            raise ValueError("Wrong eps_grid type")

    return epsinv, eps2
    
def get_conv(dN,s_in,G, batch_eps):
    ''' Attain convolution matrix
    dN = 1/Nx/Ny
    s_in: np.array of length Nx*Ny
    G: shape (nG,2), 2 for Lk1,Lk2
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    nG,_ = G.shape
    sfft = bd.fft2(s_in)*dN

    ix = torch.arange(nG)
    ii,jj = bd.meshgrid(ix,ix,indexing='ij')
    if batch_eps:
        s_out = sfft[:, G[ii,0]-G[jj,0], G[ii,1]-G[jj,1]]
    else:
        s_out = sfft[G[ii,0]-G[jj,0], G[ii,1]-G[jj,1]]
    
    return s_out

def get_fft(dN,s_in,G):
    '''
    FFT to get Fourier components
    
    s_in: np.2d array of size (Nx,Ny)
    G: shape (nG,2), 2 for Gx,Gy
    s_out: 1/N sum a_m exp(-2pi i mk/n), shape (nGx*nGy)
    '''
    
    sfft = bd.fft2(s_in)*dN
    return sfft[G[:,0],G[:,1]]


def get_ifft(Nx,Ny,s_in,G):
    '''
    Reconstruct real-space fields
    '''
    dN = 1./Nx/Ny
    nG,_ = G.shape

    s0 = bd.zeros((Nx,Ny),dtype=complex)
    for i in range(nG):
        x = G[i,0]
        y = G[i,1]

        stmp = bd.zeros((Nx,Ny),dtype=complex)
        stmp[x,y] = 1.
        s0 = s0 + s_in[i]*stmp

    s_out = bd.ifft2(s0)/dN
    return s_out
