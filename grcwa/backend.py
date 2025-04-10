import numpy as np

# Import autograd if available
try:
    import autograd.numpy as npa
    from autograd.builtins import isinstance as aisinstance
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False

# Import pytorch if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class NumpyBackend():
    """ Numpy Backend """
    isinstance = staticmethod(isinstance)
    pi = staticmethod(np.pi)
    ndarray = staticmethod(np.ndarray)

    array = staticmethod(np.array)
    sum = staticmethod(np.sum)
    vstack = staticmethod(np.vstack)
    hstack = staticmethod(np.hstack)
    zeros_like = staticmethod(np.zeros_like)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    meshgrid = staticmethod(np.meshgrid)
    reshape = staticmethod(np.reshape)
    where = staticmethod(np.where)
    concatenate = staticmethod(np.concatenate)
    eye = staticmethod(np.eye)
    diag = staticmethod(np.diag)        
    transpose = staticmethod(np.transpose)                
        
    eig = staticmethod(np.linalg.eig)
    inv = staticmethod(np.linalg.inv)
    dot = staticmethod(np.dot)
    outer = staticmethod(np.outer)
    conj = staticmethod(np.conj)
    trace = staticmethod(np.trace)
    fft2 = staticmethod(np.fft.fft2)
    ifft2 = staticmethod(np.fft.ifft2)

    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    exp = staticmethod(np.exp)
    sqrt = staticmethod(np.sqrt)
    real = staticmethod(np.real)
    imag = staticmethod(np.imag)
    abs = staticmethod(np.abs)    

    range = staticmethod(range)

if AG_AVAILABLE:
    from .primitives import (eig, inv)    
    class AutogradBackend():
        """ Autograd Backend """
        isinstance = staticmethod(aisinstance)
        pi = staticmethod(npa.pi)
        ndarray = staticmethod(npa.ndarray)
        
        array = staticmethod(npa.array)
        sum = staticmethod(npa.sum)
        vstack = staticmethod(npa.vstack)
        hstack = staticmethod(npa.hstack)
        zeros_like = staticmethod(npa.zeros_like)
        zeros = staticmethod(npa.zeros)
        ones = staticmethod(npa.ones)
        meshgrid = staticmethod(npa.meshgrid)
        reshape = staticmethod(npa.reshape)
        where = staticmethod(npa.where)
        concatenate = staticmethod(npa.concatenate)
        eye = staticmethod(npa.eye)
        diag = staticmethod(npa.diag)        
        transpose = staticmethod(npa.transpose)                
        
        eig = staticmethod(eig)
        inv = staticmethod(inv)
        dot = staticmethod(npa.dot)
        outer = staticmethod(npa.outer)
        conj = staticmethod(npa.conj)
        trace = staticmethod(npa.trace)
        fft2 = staticmethod(npa.fft.fft2)
        ifft2 = staticmethod(npa.fft.ifft2)

        sin = staticmethod(npa.sin)
        cos = staticmethod(npa.cos)
        exp = staticmethod(npa.exp)
        sqrt = staticmethod(npa.sqrt)
        real = staticmethod(npa.real)
        imag = staticmethod(npa.imag)
        abs = staticmethod(npa.abs)

        range = staticmethod(range)

import torch

if TORCH_AVAILABLE:
    def torch_transpose(tensor):
        return tensor.T

    class TorchBackend():
        """ PyTorch Backend """
        isinstance = staticmethod(isinstance)
        pi = staticmethod(torch.pi)
        tensor = staticmethod(torch.tensor)

        array = staticmethod(torch.tensor)
        sum = staticmethod(torch.sum)
        vstack = staticmethod(torch.vstack)
        hstack = staticmethod(torch.hstack)
        zeros_like = staticmethod(torch.zeros_like)
        zeros = staticmethod(torch.zeros)
        ones = staticmethod(torch.ones)
        meshgrid = staticmethod(torch.meshgrid)
        reshape = staticmethod(torch.reshape)
        where = staticmethod(torch.where)
        concatenate = staticmethod(torch.cat)
        eye = staticmethod(torch.eye)
        diag = staticmethod(torch.diag)        
        transpose = staticmethod(torch_transpose)                
            
        eig = staticmethod(torch.linalg.eig)
        inv = staticmethod(torch.inverse)
        dot = staticmethod(torch.matmul)  # Use matmul for dot product
        outer = staticmethod(torch.ger)   # Use ger for outer product
        conj = staticmethod(torch.conj)
        trace = staticmethod(torch.trace)
        fft2 = staticmethod(torch.fft.fft2)
        ifft2 = staticmethod(torch.fft.ifft2)

        sin = staticmethod(torch.sin)
        cos = staticmethod(torch.cos)
        exp = staticmethod(torch.exp)
        sqrt = staticmethod(torch.sqrt)
        real = staticmethod(torch.real)
        imag = staticmethod(torch.imag)
        abs = staticmethod(torch.abs)

        range = staticmethod(torch.arange)

backend = NumpyBackend()

def set_backend(name):
    """
    Set the backend for the simulations.
    This function monkey-patches the backend object by changing its class.
    This way, all methods of the backend object will be replaced.
    
    Parameters
    ----------
    name : {'numpy', 'autograd'}
        Name of the backend. HIPS/autograd must be installed to use 'autograd'.
    """
    # perform checks
    if name == 'autograd' and not AG_AVAILABLE:
        raise ValueError("Autograd backend is not available, autograd must \
            be installed.")
    elif name == 'torch' and not TORCH_AVAILABLE:
        raise ValueError("Pytorch backend is not available, pytorch must \
            be installed.")

    # change backend by monkeypatching
    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'autograd':
        backend.__class__ = AutogradBackend
    elif name =='torch':
        backend.__class__ = TorchBackend
    else:
        raise ValueError("unknown backend")
