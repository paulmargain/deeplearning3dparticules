#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:00:38 2020

@author: paulmargain
"""
import numpy as np
import matplotlib.pyplot as plt

###Initialisations
Nx, Ny, Nz = 128,128,128
Nx_ext, Ny_ext, Nz_ext = 256,256,128
nb = 1.333
wl = 406
pi=3.141592
NA=1.4
i=complex(0,1)
drx, dry, drz = 100,100,100
FPDist = -Nz*drz*0.5
mod_dist=0
dkx_ext = 2*pi/(Nx_ext*drx)
dky_ext = 2*pi/(Ny_ext*dry)
kx_ext = dkx_ext * (np.array([k for k in range(Nx_ext//2)] + [ k for k in range (-Nx_ext//2,0)]))
ky_ext = dky_ext * (np.array([k for k in range(Ny_ext//2)] + [ k for k in range (-Ny_ext//2,0)]))
[Kyy_ext,Kxx_ext] = np.meshgrid(ky_ext**2, kx_ext**2)
k = 2*pi/wl*nb
k0 = 2*pi/wl        
dphi_ext = np.real(k - np.sqrt((k**2 - Kxx_ext - Kyy_ext),dtype="complex128")) #diffraction phase factor extended
x = nb* np.ones((Nx,Ny,Nz),dtype="complex128")
theta= 0;
u = np.zeros([Nx_ext,Ny_ext,Nz],dtype = "complex128") ##Volume 3D du champ propagé
uin = np.ones((Nx_ext,Ny_ext),dtype ="complex128")               
y = uin
            
for ind_z in range (0,Nz):


    y = np.fft.ifft2(np.fft.fft2(y)*np.exp(-i*dphi_ext*drz)) #Diffraction step
    upad=np.ones((Nx_ext,Ny_ext),dtype ="complex128")  
    upad[(Nx_ext-Nx)//2:(Nx_ext+Nx)//2,(Ny_ext-Ny)//2:(Ny_ext+Ny)//2] = np.exp(i*k0*x[:,:,ind_z]*drz/np.cos(theta))
    y=y*upad
    u[:,:,ind_z] = y

#pupil 1,2,3 correspond to tree different pupil functions
 
pupil1,pupil2,pupil3=np.zeros((256,256),dtype ="complex128"),np.zeros((256,256),dtype ="complex128"),np.zeros((256,256),dtype ="complex128")
for a in range (256):
    for b in range(256):
        
        if np.sqrt(kx_ext[max(127-a,a-127)]**2+ky_ext[max(127-b,b-127)]**2)<= NA/wl:
            pupil1[a][b]=1
            pupil2[a][b]=np.exp(i*k0*kx_ext[a]**2)
            pupil3[a][b]=np.exp(i*k0*kx_ext[a]**2+ky_ext[b]**2)
            
        else:
            pupil1[a][b]=0
            pupil2[a][b]=0
            pupil3[a][b]=0


y0 = np.fft.ifft2(np.fft.fft2(y)*np.exp(-i*dphi_ext*FPDist)) #overflow ici ?
y1 = np.fft.ifft2(np.fft.fft2(y)*np.exp(-i*dphi_ext*FPDist)*pupil1) 
y2 = np.fft.ifft2(np.fft.fft2(y)*np.exp(-i*dphi_ext*FPDist)*pupil2) 
y3 = np.fft.ifft2(np.fft.fft2(y)*np.exp(-i*dphi_ext*FPDist)*pupil3) 
                
y0 = y0*np.exp(i*k*mod_dist) #BPM original plan is missing this global phase, can be optional
y1 = y1*np.exp(i*k*mod_dist) 
y2 = y2*np.exp(i*k*mod_dist) 
y3 = y3*np.exp(i*k*mod_dist) 

y0 = y0[Nx//2:Nx_ext-Nx//2,Ny//2:Ny_ext-Ny//2 ] #Select the central image
y1 = y1[Nx//2:Nx_ext-Nx//2,Ny//2:Ny_ext-Ny//2 ] #Select the central image
y2 = y2[Nx//2:Nx_ext-Nx//2,Ny//2:Ny_ext-Ny//2 ] #Select the central image
y3 = y3[Nx//2:Nx_ext-Nx//2,Ny//2:Ny_ext-Ny//2 ] #Select the central image
     
## Affichage des pupil functions  
plt.figure()

# pupil 1
plt.subplot(231)
plt.imshow(np.abs(pupil1), cmap='gray')
plt.title('module pupil1')

plt.subplot(234)
plt.imshow(np.angle(pupil1), cmap='gray')
plt.title('phase pupil1')

# pupil 2
plt.subplot(232)
plt.imshow(np.abs(pupil2), cmap='gray')
plt.title('module pupil2')

plt.subplot(235)
plt.imshow(np.angle(pupil2), cmap='gray')
plt.title('phase pupil2')
# pupil 3
plt.subplot(233)
plt.imshow(np.abs(pupil3), cmap='gray')
plt.title('module pupil3')

plt.subplot(236)
plt.imshow(np.angle(pupil3), cmap='gray')
plt.title('phase pupil3')
plt.show()
## Affichage des y

plt.figure()
# y0
plt.subplot(241)
plt.imshow(np.abs(y0), cmap='gray', origin='upper')
plt.title('module y0')

plt.subplot(245)
plt.imshow(np.angle(y0), cmap='gray')
plt.title('phase y0')

# y1
plt.subplot(242)
plt.imshow(np.abs(y1), cmap='gray')
plt.title('module y1')

plt.subplot(246)
plt.imshow(np.angle(y1), cmap='gray')
plt.title('phase y1')
# y2
plt.subplot(243)
plt.imshow(np.abs(y2), cmap='gray')
plt.title('module y2')

plt.subplot(247)
plt.imshow(np.angle(y2), cmap='gray')
plt.title('phase y2')

# y3
plt.subplot(244)
plt.imshow(np.abs(y3), cmap='gray')
plt.title('module y3')

plt.subplot(248)
plt.imshow(np.angle(y3), cmap='gray')
plt.title('phase y3')
plt.show()


