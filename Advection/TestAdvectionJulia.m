clear

upwind = 1

xmin     = -0.0;  xmax    =    1; Lx = xmax - xmin;
ymin     = -0.0;  ymax    =    1; Ly = ymax - ymin;

zmin     = -0.00; zmax    =    1; Lz = zmax - zmin;
nx       = 4*32;
ny       = 4*32;
nz       = 4*32;

Vx      =  ones(nx+1,ny+0,nz+0);
Vy      =  zeros(nx+0,ny+1,nz+0);
Vz      =  zeros(nx+0,ny+0,nz+1);

dx       = Lx/nx;                                                       
dy       = Ly/ny;
dz       = Lz/nz;
dt       = 0.25*min(min(dx,dy),dz) / 1


xce=xmin-dx/2:dx:xmax+dx/2;
yce=ymin-dy/2:dy:ymax+dy/2;
zce=zmin-dz/2:dz:zmax+dz/2;

xc=xmin+dx/2:dx:xmax-dx/2;
yc=ymin+dy/2:dy:ymax-dy/2;

[xce2,yce2,zce2] = ndgrid(xce,yce,zce);

xC = 0.1;
yC = 0.5*(ymin+ymax);
zC = 0.5*(zmin+zmax);

Tc_ex = exp(-(xce2-xC).^2/ 0.001 - (yce2-yC).^2/ 0.001 - (zce2-zC).^2/ 0.001);

max(Tc_ex(:))

vxm1   = min(Vx(1:end-1,:,int16(ceil(nz/2))),0);
vxp1   = max(Vx(2:end  ,:,int16(ceil(nz/2))),0);
vym1   = min(Vy(:,1:end-1,int16(ceil(nz/2))),0);
vyp1   = max(Vy(:,2:end  ,int16(ceil(nz/2))),0);

phi1     = Tc_ex(2:end-1,2:end-1,int16(ceil(nz/2)));
order    = 2;
dimsplit = 1;



time = 0;

for it=1:1
    
    time = time + dt;

    % 1.1 Upwind
    phi1o = phi1;
    phi1t = phi1;
    for step=1:order
        if upwind == 1
%             phie  = [1*phi1t(end,:); phi1t; 1*phi1t(1,:)]; % periodic x
%             phie  = [1*phie(:,end)  phie  1*phie(:,1)];
            phie  = [1*phi1t(1,:); phi1t; 1*phi1t(end,:)]; % no flux
            phie  = [1*phie(:,1)  phie  1*phie(:,end)];
            phxm1 = 1/dx*(phie(2:end-1,2:end-1) - phie(1:end-2,2:end-1));
            phxp1 = 1/dx*(phie(3:end,2:end-1)   - phie(2:end-1,2:end-1));
        else
            [phxm1,phxp1,c,d] = weno5(phi1t, dx, dy , 1, 0, 1);
        end
        phi1t = phi1t - dimsplit*dt*(vxp1.*phxm1 + vxm1.*phxp1);
    end
    phi1t = (1/order)*phi1t + (1-1/order)*phi1o;
    phi1o = phi1t;
    for step=1:order
        if upwind == 1
            phie  = [1*phi1t(1,:); phi1t; 1*phi1t(end,  :)]; % no flux
            phie  = [1*phie(:  ,1)  phie  1*phie(:,end)];
            phym1 = 1/dy*(phie(2:end-1,2:end-1) - phie(2:end-1,1:end-2));
            phyp1 = 1/dy*(phie(2:end-1,3:end)   - phie(2:end-1,2:end-1));
        else
            [a,b,phym1,phyp1] = weno5(phi1t, dx, dy , 1, 0, 1);
        end
        phi1t = phi1t - dt*(vyp1.*phym1 + vym1.*phyp1) - (1-dimsplit)*dt*(vxp1.*phxm1 + vxm1.*phxp1);
    end
    phi1 = (1/order)*phi1t + (1-1/order)*phi1o;
    
end

max(phi1(:))

centerx = time*1 + 0.1

figure(1),clf;
imagesc(xc,yc, phi1' )
colorbar
set(gca,'ydir','normal')
