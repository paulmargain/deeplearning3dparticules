Nx = 128;
Ny = 128;
Nz = 128;
Nx_ext = 256;
Ny_ext = 256;
Nz_ext = 256;
nb = 1.333;
wl = 406;
drx = 100;
dry = 100;
drz = 100;
FPDist = -Nz*drz;
mod_dist=0;
dkx_ext = 2*pi/(Nx_ext*drx);
dky_ext = 2*pi/(Ny_ext*dry);
kx_ext = (dkx_ext * [0:Nx_ext/2-1, -Nx_ext/2:-1]');
ky_ext = (dky_ext * [0:Ny_ext/2-1, -Ny_ext/2:-1]');
[Kyy_ext,Kxx_ext] = meshgrid(ky_ext.^2, kx_ext.^2);
k = 2*pi/wl*nb;
k0 = 2*pi/wl; 
dphi_ext = real(k - sqrt(k^2 - Kxx_ext - Kyy_ext));% diffraction phase factor extended
x = nb* ones(Nx,Ny,Nz);
theta= 0;
u = zeros([Nx_ext,Ny_ext,Nz]);%Volume 3D du champ propagé
uin = ones(Nx_ext,Ny_ext);                
uouthat = uin;
                for ind_z = 1:Nz
             
                    uouthat = ifft2(fft2(uouthat).*exp(-1i*dphi_ext*drz));%Diffraction step
                   
                  
                    uouthat = uouthat.*padarray(exp(1i*k0*x(:,:,ind_z)*drz/cos(theta)),...
                        [(Nx_ext - Nx)/2,(Ny_ext - Ny)/2],1,'both');%Refraction step
                    u(:,:,ind_z) = uouthat;
                   
                end
                
                
                uouthat = ifft2(fft2(uouthat).*exp(-1i*dphi_ext*FPDist));%Free-space propagation until the focal plane
                
               
                uouthat = uouthat*exp(1i*k*mod_dist);%BPM original plan is missing this global phase, can be optional
                
                uouthat = uouthat(1 + end/2 - Nx/2:end/2 + Nx/2,1 + end/2 - Ny/2:end/2 + Ny/2);%Select the central image
