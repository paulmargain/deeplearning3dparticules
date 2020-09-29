#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:00:38 2020

@author: paulmargain
"""
import numpy as np

###Initialisations
Nx, Ny, Nz = 128,128,128
Nx_ext, Ny_ext, Nz_ext = 256,256,256
nb = 1.333
wl = 406
pi=3.141592
i=complex(0,1)
drx, dry, drz = 100,100,100
FPDist = -Nz*drz
mod_dist=0
dkx_ext = 2*pi/(Nx_ext*drx)
dky_ext = 2*pi/(Ny_ext*dry)
kx_ext = dkx_ext * (np.array([k for k in range(Nx_ext//2)] + [ k for k in range (-Nx_ext//2,0)]))
ky_ext = dky_ext * (np.array([k for k in range(Ny_ext//2)] + [ k for k in range (-Ny_ext//2,0)]))
[Kyy_ext,Kxx_ext] = np.meshgrid(ky_ext**2, kx_ext**2)
k = 2*pi/wl*nb
k0 = 2*pi/wl        
dphi_ext = np.real(k - np.sqrt((k**2 - Kxx_ext - Kyy_ext),dtype="complex_")) #diffraction phase factor extended
x = nb* np.ones((Nx,Ny,Nz),dtype="complex_")
theta= 0;
u = np.zeros([Nx_ext,Ny_ext,Nz],dtype = "complex_") ##Volume 3D du champ propagé
uin = np.ones((Nx_ext,Ny_ext),dtype ="complex_")               
uouthat = uin
            
for ind_z in range (0,Nz):


    uouthat = np.fft.ifft2(np.fft.fft2(uouthat)*np.exp(-i*dphi_ext*drz)) #Diffraction step
    upad=uin
    upad[(Nx_ext-Nx)//2:(Nx_ext+Nx)//2,(Ny_ext-Ny)//2:(Ny_ext+Ny)//2] = np.exp(i*k0*x[:,:,ind_z]*drz/np.cos(theta))
    uouthat=uouthat*upad
    u[:,:,ind_z] = uouthat
    
                
               
uouthat = np.fft.ifft2(np.fft.fft2(uouthat)*np.exp(-i*dphi_ext*FPDist)) #Free-space propagation until the focal plane
                
        
uouthat = uouthat*np.exp(i*k*mod_dist) #BPM original plan is missing this global phase, can be optional


uouthat = uouthat[Nx//2:Nx_ext-Nx//2,Ny//2:Ny_ext-Ny//2 ] #Select the central image
       

