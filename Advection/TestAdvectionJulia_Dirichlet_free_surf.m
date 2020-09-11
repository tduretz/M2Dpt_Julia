clear
% Main switches
WENO5    = 0;
test     = 4;
vstep    = 5;
order    = 2;
dimsplit = 1;
% Space
xmin     = -0.5;  xmax    =    0.5; Lx = xmax - xmin;
ymin     = -0.5;  ymax    =    0.5; Ly = ymax - ymin;
zmin     = -0.50; zmax    =    0.5; Lz = zmax - zmin;
nx       = 4*32;
ny       = 4*32;
nz       = 1;
nt       = 1000;
nout     = 20;
dx       = Lx/nx;
dy       = Ly/ny;
dz       = Lz/nz;
dt       = 0.25*min(min(dx,dy),dz) / 4;

% Dirichlet boundary conditions
BC   = 4;
valW = 2.0;
valE = 1.0;
valS = 1.0;
valN = 1.0;
valX = 1.0; % surface value

% Mesh
xce=xmin-dx/2:dx:xmax+dx/2;
yce=ymin-dy/2:dy:ymax+dy/2;
zce=zmin-dz/2:dz:zmax+dz/2;
xc=xmin+dx/2:dx:xmax-dx/2;
yc=ymin+dy/2:dy:ymax-dy/2;
zc=zmin+dz/2:dz:zmax-dz/2;
xv=xmin:dx:xmax;
yv=ymin:dy:ymax;
[xce2,yce2,zce2] = ndgrid(xce,yce,zce);
[xvx2,~,~] = ndgrid(xc,yv,zc);
[~,yvy2,~] = ndgrid(xv,yc,zc);
% Storage
maxphiv = zeros(nt,1);
% Setups
if test==0
    % Shift in x
    xC = 0.0;
    yC = 0.0;
    zC = 0.5*(zmin+zmax);
    Vx      =  ones(nx+1,ny+0,nz+0);
    Vy      =  zeros(nx+0,ny+1,nz+0);
    Vz      =  zeros(nx+0,ny+0,nz+1);
elseif test==1
    % Shift in y
    xC = 0.0;
    yC = 0.0;
    zC = 0.5*(zmin+zmax);
    Vx      =  zeros(nx+1,ny+0,nz+0);
    Vy      =   ones(nx+0,ny+1,nz+0);
    Vz      =  zeros(nx+0,ny+0,nz+1);
elseif test==3
    % Rotate
    xC = 0.0;
    yC = 0.3;
    zC = 0.5*(zmin+zmax);
    Vx      =   xvx2';
    Vy      =  -yvy2';
    Vz      =  zeros(nx+0,ny+0,nz+1);
elseif test==4
    % Shift in x y
    xC = 0.0;
    yC = 0.0;
    zC = 0.5*(zmin+zmax);
    Vx      =   ones(nx+1,ny+0,nz+0);
    Vy      =  -ones(nx+0,ny+1,nz+0);
    Vz      =  zeros(nx+0,ny+0,nz+1);
end
% Initial condition
% Tc_ex = exp(-(xce2-xC).^2/ 0.001 - (yce2-yC).^2/ 0.001 - (zce2-zC).^2/ 0.001);
Tc_ex = exp(-(xce2-xC).^2/ 0.001 - (yce2-yC).^2/ 0.001);
% Pre-process
vxm1  = min(Vx(1:end-1,:,int16(ceil(nz/2))),0);
vxp1  = max(Vx(2:end  ,:,int16(ceil(nz/2))),0);
vym1  = min(Vy(:,1:end-1,int16(ceil(nz/2))),0);
vyp1  = max(Vy(:,2:end  ,int16(ceil(nz/2))),0);
phi1  = Tc_ex(2:end-1,2:end-1,int16(ceil(nz/2)));
VxC   = 0.5*(Vx(1:end-1,:,:)+Vx(2:end-0,:,:));
VyC   = 0.5*(Vy(:,1:end-1,:)+Vy(:,2:end-0,:));

% Draw top free surface
xc2 = xce2(2:end-1,2:end-1,2:end-1);
yc2 = yce2(2:end-1,2:end-1,2:end-1);
zc2 = zce2(2:end-1,2:end-1,2:end-1);
in    = ones(size(phi1));
in(yc2>0.25*xc2+0.25) = 0;
phi1(in==0) = -1;
inC = in==1;
bc  = 5*ones(nx+2,ny+2);
bc(2:end-1,2:end-1) = inC;
bc(bc==1) = -1;
bc(1,:) = 1; bc(end,:) = 2;
bc(:,1) = 3; bc(:,end) = 4;







% figure(1)
% hold on
% % imagesc(xc,yc, bc' ), axis xy image, colorbar
% imagesc(xce,yce, phiex' ), axis xy image, colorbar
% 
% drawnow






time  = 0;
%% Time loop
for it=1:nt
    
    time = time + dt;
        
    % 1.1 Upwind
    phi1o = phi1;
    phi1t = phi1;
    for step=1:order
        if WENO5 == 0 % upwind
            
            phiex  = -0*ones(nx+2,ny+2);

            % Extended values in x - NEW FOR SURFACE
            for j=1:ny+2
                for i=1:nx+2

                    if bc(i,j)==1 && bc(i+1,j)==-1 && i==1
                         phiex(i,j) = 2*valW - phi1t(i-1+1,j-1);
                    elseif bc(i,j)==2 && bc(i-1,j)==-1 && i==nx+2
                         phiex(i,j) = 2*valE - phi1t(i-1-1,j-1);
                    elseif bc(i,j)==0 && (bc(i+1,j)==-1)
                         phiex(i,j) = 2*valX - phi1t(i-1+1,j-1);
                    elseif bc(i,j)==0 && (bc(i-1,j)==-1)
                         phiex(i,j) = 2*valX - phi1t(i-1-1,j-1);
                    elseif i>1 && i<nx+2 && j>1 && j<ny+2
                        phiex(i,j) = phi1t(i-1,j-1);
                    end

                end
            end

%             if BC == 0 - OLD STANDARD BCs
%                 phie  = [1*phi1t(end,:); phi1t; 1*phi1t(1,:)]; % periodic x
%                 phie  = [1*phie(:,end)  phie  1*phie(:,1)];
%             elseif BC == 4
%                 phie  = [2*valW-phi1t(1,:); phi1t; 2*valE-phi1t(end,:)]; 
%                 phie  = [2*valS-phie(:,1)  phie  2*valN-phie(:,end)];
%             end
            phxm1 = 1/dx*(phiex(2:end-1,2:end-1) - phiex(1:end-2,2:end-1));
            phxp1 = 1/dx*(phiex(3:end,2:end-1)   - phiex(2:end-1,2:end-1));
        else
            [phxm1,phxp1,c,d] = weno5_v3(phi1t, dx, dy , 1, 0, BC, valW, valE, valS, valN);
        end
        phi1t = phi1t - dimsplit*dt*(vxp1.*phxm1 + vxm1.*phxp1);
    end
    phi1t(inC) = (1/order)*phi1t(inC) + (1-1/order)*phi1o(inC);
    phi1o = phi1t;
    for step=1:order
        if WENO5 == 0 % upwind
            
            phiey  = -0*ones(nx+2,ny+2);
            
            % Extended values in y - NEW FOR SURFACE
            for j=1:ny+2
                for i=1:nx+2

                    if bc(i,j)==3 && bc(i,j+1)==-1 && j==1
                         phiey(i,j) = 2*valS - phi1t(i-1,j-1+1);
                    elseif bc(i,j)==4 && bc(i,j-1)==-1 && j==ny+2
                         phiey(i,j) = 2*valN - phi1t(i-1,j-1-1);
                    elseif bc(i,j)==0 && (bc(i,j+1)==-1)
                         phiey(i,j) = 2*valX - phi1t(i-1,j-1+1);
                    elseif bc(i,j)==0 && (bc(i,j-1)==-1)
                         phiey(i,j) = 2*valX - phi1t(i-1,j-1-1);
                    elseif i>1 && i<nx+2 && j>1 && j<ny+2
                        phiey(i,j) = phi1t(i-1,j-1);
                    end

                end
            end
            
%            if BC == 0 - OLD STANDARD BCs
%                 phie  = [1*phi1t(end,:); phi1t; 1*phi1t(1,:)]; % periodic x
%                 phie  = [1*phie(:,end)  phie  1*phie(:,1)];
%             elseif BC == 4
%                 phie  = [2*valW-phi1t(1,:); phi1t; 2*valE-phi1t(end,:)];
%                 phie  = [2*valS-phie(:,1)  phie  2*valN-phie(:,end)];
%             end
            phym1 = 1/dy*(phiey(2:end-1,2:end-1) - phiey(2:end-1,1:end-2));
            phyp1 = 1/dy*(phiey(2:end-1,3:end)   - phiey(2:end-1,2:end-1));
        else
            [a,b,phym1,phyp1] = weno5_v3(phi1t, dx, dy , 0, 1, BC, valW, valE, valS, valN);
        end
        phi1t = phi1t - dt*(vyp1.*phym1 + vym1.*phyp1) - (1-dimsplit)*dt*(vxp1.*phxm1 + vxm1.*phxp1);
    end
    phi1(inC) = (1/order)*phi1t(inC) + (1-1/order)*phi1o(inC);
    % Post-processing
    % centerx = time*1 + 0.1
    maxphiv(it) = max(max(phi1(:)));
    if mod(it,nout)==0
        figure(1),clf;
        hold on
        imagesc(xc,yc, phi1' )
        colorbar
        quiver(xc(1:vstep:end,1:vstep:end),yc(1:vstep:end,1:vstep:end),VxC(1:vstep:end,1:vstep:end,1:vstep:end,1:vstep:end)',VyC(1:vstep:end,1:vstep:end,1:vstep:end,1:vstep:end)', 'w')
        set(gca,'ydir','normal')
%         caxis([0 2])
        title(maxphiv(it))
        drawnow
        figure(2),clf;
        plot(1:it, maxphiv(1:it), 'ok')
    end
end
