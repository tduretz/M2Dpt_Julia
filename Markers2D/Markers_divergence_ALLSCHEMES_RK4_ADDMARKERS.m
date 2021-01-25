% Solving Stokes and continuity eq.
% in primitive variable formulation
% with variable viscosity
% using FD with staggered grid

% Clearing memory and figures
clear all;

% Advection scheme: 
% 1=linear; 2=(Jenny, 2001); 3=(Gerya,2019); 4=(Gerya, 2020) 


% 1) Define Numerical model
xsize=100; % Horizontal model size, m
ysize=100; % Vertical model size, m
Nx=41; % Horizontal grid resolution
Ny=41; % Vertical grid resolution
Nx1=Nx+1;
Ny1=Ny+1;
dx=xsize/(Nx-1); % Horizontal grid step, m
dy=ysize/(Ny-1); % Vertical grid step, m
x=0:dx:xsize; % Horizontal coordinates of basic grid points, m
y=0:dy:ysize; % Vertical coordinates of basic grid points, m
xvx=0:dx:xsize; % Horizontal coordinates of vx grid points, m
yvx=-dy/2:dy:ysize+dy/2; % Vertical coordinates of vx grid points, m
xvy=-dx/2:dx:xsize+dx/2; % Horizontal coordinates of vy grid points, m
yvy=0:dy:ysize; % Vertical coordinates of vy grid points, m
xp=-dx/2:dx:xsize+dx/2; % Horizontal coordinates of P grid points, m
yp=-dy/2:dy:ysize+dy/2; % Vertical coordinates of P grid points, m
vx=zeros(Ny1,Nx1); % Vx, m/s
vy=zeros(Ny1,Nx1); % Vy, m/s
vxp=zeros(Ny1,Nx1); % Vx in pressure nodes, m/s
vyp=zeros(Ny1,Nx1); % Vy in pressure nodes, m/s
pr=zeros(Ny1,Nx1); % Pressure, Pa
gy=0; % Gravity acceleration, m/s^2
RHO=zeros(Ny,Nx); % Density, kg/m^3
ETA=zeros(Ny,Nx); % Viscosity, Pa*s

% Define markers
Nxmc=5; % Number of markers per cell in x-direction
Nymc=5; % Number of markers per cell in x-direction
Nxm=(Nx-1)*Nxmc; % Marker grid resolution in horizontal direction
Nym=(Ny-1)*Nymc; % Marker grid resolution in vertical direction
dxm=xsize/Nxm; % Marker grid step in horizontal direction,m
dym=ysize/Nym; % Marker grid step in vertical direction,m
marknum=Nxm*Nym; % Number of markers
xm=zeros(1,marknum); % Horizontal coordinates, m
ym=zeros(1,marknum); % Vertical coordinates, m
rhom=zeros(1,marknum); % Density, kg/m^3
etam=zeros(1,marknum); % Viscosity, Pa*s

% Compose density array on markers
rp=20000; % Plume radius, m
m=1; % Marker counter
for jm=1:1:Nxm
    for im=1:1:Nym
        % Define marker coordinates
        xm(m)=dxm/2+(jm-1)*dxm+(rand-0.5)*dxm;
        ym(m)=dym/2+(im-1)*dym+(rand-0.5)*dym;
        % Marker properties
        if(xm(m)<ym(m))
            rhom(m)=3300; % Mantle density
            etam(m)=1e+21; % Mantle viscosity
            tm(m)=1;
        else
            rhom(m)=3200; % Plume density
            etam(m)=1e+21; % Plume viscosity
            tm(m)=2;
        end
        % Update marker counter
        m=m+1;
    end
end

% Introducing scaled pressure
pscale=1e+21/dx;


% 2) Define global matrixes L(), R()
N=Nx1*Ny1*3; % Global number of unknowns
L=sparse(N,N); % Matrix of coefficients (left part)
R=zeros(N,1); % Vector of right parts

% Boundary conditions: free slip=-1; No Slip=1
bcleft=-1;
bcright=-1;
bctop=-1;
bcbottom=-1;

% Timestepping
dt=0e12; % initial timestep
dxymax=0.1; % max marker movement per timestep
visstep=500; % Number of steps between visualization
ntimesteps=100000; % Number of advection time steps

% Interpolate RHO, ETA from markers
RHOSUM=zeros(Ny,Nx);
ETASUM=zeros(Ny,Nx);
WTSUM=zeros(Ny,Nx);
for m=1:1:marknum
    % Define i,j indexes for the upper left node
    j=fix((xm(m)-x(1))/dx)+1;
    i=fix((ym(m)-y(1))/dy)+1;
    if(j<1)
        j=1;
    elseif(j>Nx-1)
        j=Nx-1;
    end
    if(i<1)
        i=1;
    elseif(i>Ny-1)
        i=Ny-1;
    end
    % Compute distances
    dxmj=xm(m)-x(j);
    dymi=ym(m)-y(i);
    % Compute weights
    wtmij=(1-dxmj/dx)*(1-dymi/dy);
    wtmi1j=(1-dxmj/dx)*(dymi/dy);    
    wtmij1=(dxmj/dx)*(1-dymi/dy);
    wtmi1j1=(dxmj/dx)*(dymi/dy);
    % Update properties
    % i,j Node
    RHOSUM(i,j)=RHOSUM(i,j)+rhom(m)*wtmij;
    ETASUM(i,j)=ETASUM(i,j)+etam(m)*wtmij;
    WTSUM(i,j)=WTSUM(i,j)+wtmij;
    % i+1,j Node
    RHOSUM(i+1,j)=RHOSUM(i+1,j)+rhom(m)*wtmi1j;
    ETASUM(i+1,j)=ETASUM(i+1,j)+etam(m)*wtmi1j;
    WTSUM(i+1,j)=WTSUM(i+1,j)+wtmi1j;
    % i,j+1 Node
    RHOSUM(i,j+1)=RHOSUM(i,j+1)+rhom(m)*wtmij1;
    ETASUM(i,j+1)=ETASUM(i,j+1)+etam(m)*wtmij1;
    WTSUM(i,j+1)=WTSUM(i,j+1)+wtmij1;
    % i+1,j+1 Node
    RHOSUM(i+1,j+1)=RHOSUM(i+1,j+1)+rhom(m)*wtmi1j1;
    ETASUM(i+1,j+1)=ETASUM(i+1,j+1)+etam(m)*wtmi1j1;
    WTSUM(i+1,j+1)=WTSUM(i+1,j+1)+wtmi1j1;
end
% Compute ETA, RHO
for j=1:1:Nx
    for i=1:1:Ny
        if(WTSUM(i,j)>0)
            RHO(i,j)=RHOSUM(i,j)/WTSUM(i,j);
            ETA(i,j)=ETASUM(i,j)/WTSUM(i,j);
        end
    end
end

% Compute viscosity in pressure nodes
ETAP=zeros(Ny1,Nx1); % Viscosity in pressure nodes, Pa*s
for j=2:1:Nx
    for i=2:1:Ny
        % Harmonic average
        ETAP(i,j)=1/((1/ETA(i,j)+1/ETA(i,j-1)+...
              1/ETA(i-1,j)+1/ETA(i-1,j-1))/4);
    end
end

% 3) Composing global matrixes L(), R() for OMEGA
% Going through all points of the 2D grid and
% composing respective equations
for j=1:1:Nx1
    for i=1:1:Ny1
        % Define global indexes in algebraic space
        kvx=((j-1)*Ny1+i-1)*3+1; % Vx
        kvy=kvx+1; % Vy
        kpm=kvx+2; % P
        
        % Vx equation External points
        if(i==1 || i==Ny1 || j==1 || j==Nx || j==Nx1)
            % Boundary Condition
            % 1*Vx=0
            L(kvx,kvx)=1; % Left part
            R(kvx)=0; % Right part
            % Top boundary
            if(i==1 && j>1 && j<Nx)
                L(kvx,kvx+3)=bctop; % Left part
            end
            % Bottom boundary
            if(i==Ny1 && j>1 && j<Nx)
                L(kvx,kvx-3)=bcbottom; % Left part
            end
        % Internal BC on Diagonal Lines
        elseif(i>3 && i<Ny1-2 && (j==i-0 || j==i+2))
            % 1*Vx=0
            L(kvx,kvx)=1; % Left part
            if(j==i-0)
                R(kvx)=1e-9; % Right part
            else
                R(kvx)=-0e-9; % Right part
            end
        else
        % Internal points: x-Stokes eq.
        % ETA*(d2Vx/dx^2+d2Vx/dy^2)-dP/dx=0
        %            Vx2
        %             |
        %        Vy1  |  Vy3
        %             |
        %     Vx1-P1-Vx3-P2-Vx5
        %             |
        %        Vy2  |  Vy4
        %             |
        %            Vx4
        %
        % Viscosity points
        ETA1=ETA(i-1,j);
        ETA2=ETA(i,j);
        ETAP1=ETAP(i,j);
        ETAP2=ETAP(i,j+1);
        % Left part
        L(kvx,kvx-Ny1*3)=2*ETAP1/dx^2; % Vx1
        L(kvx,kvx-3)=ETA1/dy^2; % Vx2
        L(kvx,kvx)=-2*(ETAP1+ETAP2)/dx^2-(ETA1+ETA2)/dy^2; % Vx3
        L(kvx,kvx+3)=ETA2/dy^2; % Vx4
        L(kvx,kvx+Ny1*3)=2*ETAP2/dx^2; % Vx5
        L(kvx,kvy)=-ETA2/dx/dy;  % Vy2
        L(kvx,kvy+Ny1*3)=ETA2/dx/dy;  % Vy4
        L(kvx,kvy-3)=ETA1/dx/dy;  % Vy1
        L(kvx,kvy+Ny1*3-3)=-ETA1/dx/dy;  % Vy3
        L(kvx,kpm)=pscale/dx; % P1
        L(kvx,kpm+Ny1*3)=-pscale/dx; % P2
        % Right part
        R(kvx)=0;
        end
        
        % Vy equation External points
        if(j==1 || j==Nx1 || i==1 || i==Ny || i==Ny1)
            % Boundary Condition
            % 1*Vy=0
            L(kvy,kvy)=1; % Left part
            R(kvy)=0; % Right part
            % Left boundary
            if(j==1 && i>1 && i<Ny)
                L(kvy,kvy+3*Ny1)=bcleft; % Left part
            end
            % Right boundary
            if(j==Nx1 && i>1 && i<Ny)
                L(kvy,kvy-3*Ny1)=bcright; % Left part
            end
        % Internal BC on Diagonal Lines
        elseif(i>3 && i<Ny-2 && (j==i+1 || j==i+3))
                % 1*Vx=0
                L(kvy,kvy)=1; % Left part
                if(j==i+1)
                    R(kvy)=1e-9; % Right part
                else
                    R(kvy)=-0e-9; % Right part
                end
        else
        % Internal points: y-Stokes eq.
        % ETA*(d2Vy/dx^2+d2Vy/dy^2)-dP/dy=-RHO*gy
        %            Vy2
        %             |
        %         Vx1 P1 Vx3
        %             |
        %     Vy1----Vy3----Vy5
        %             |
        %         Vx2 P2 Vx4
        %             |
        %            Vy4
        %
        % Viscosity points
        % Viscosity points
        ETA1=ETA(i,j-1);
        ETA2=ETA(i,j);
        ETAP1=ETAP(i,j);
        ETAP2=ETAP(i+1,j);
        % Density gradients
        dRHOdx=(RHO(i,j)-RHO(i,j-1))/dx;
        dRHOdy=(RHO(i+1,j-1)-RHO(i-1,j-1)+...
                RHO(i+1,j)-RHO(i-1,j))/dy/4;
        % Left part
        L(kvy,kvy-Ny1*3)=ETA1/dx^2; % Vy1
        L(kvy,kvy-3)=2*ETAP1/dy^2; % Vy2
        L(kvy,kvy)=-2*(ETAP1+ETAP2)/dy^2-...
                      (ETA1+ETA2)/dx^2-...
                      dRHOdy*gy*dt; % Vy3
        L(kvy,kvy+3)=2*ETAP2/dy^2; % Vy4
        L(kvy,kvy+Ny1*3)=ETA2/dx^2; % Vy5
        L(kvy,kvx)=-ETA2/dx/dy-dRHOdx*gy*dt/4; %Vx3
        L(kvy,kvx+3)=ETA2/dx/dy-dRHOdx*gy*dt/4; %Vx4
        L(kvy,kvx-Ny1*3)=ETA1/dx/dy-dRHOdx*gy*dt/4; %Vx1
        L(kvy,kvx+3-Ny1*3)=-ETA1/dx/dy-dRHOdx*gy*dt/4; %Vx2
        L(kvy,kpm)=pscale/dy; % P1
        L(kvy,kpm+3)=-pscale/dy; % P2
        
        % Right part
        R(kvy)=-(RHO(i,j-1)+RHO(i,j))/2*gy;
        end
        
        % P equation External points
        if(i==1 || j==1 || i==Ny1 || j==Nx1 ||...
          (i==2 && j==2))
            % Boundary Condition
            % 1*P=0
            L(kpm,kpm)=1; % Left part
            R(kpm)=0; % Right part
            % Real BC
            if(i==2 && j==2)
                L(kpm,kpm)=1*pscale; %Left part
                R(kpm)=1e+9; % Right part
            end
        else
        % Internal points: continuity eq.
        % dVx/dx+dVy/dy=0
        %            Vy1
        %             |
        %        Vx1--P--Vx2
        %             |
        %            Vy2
        %
        % Left part
        L(kpm,kvx-Ny1*3)=-1/dx; % Vx1
        L(kpm,kvx)=1/dx; % Vx2
        L(kpm,kvy-3)=-1/dy; % Vy1
        L(kpm,kvy)=1/dy; % Vy2
        % Right part
        R(kpm)=0;
        end
        
    end
end

% 4) Solving matrixes, reloading solution
S=L\R; % Obtaining algebraic vector of solutions S()
% Reload solutions S() to vx(), vy(), p()
% Going through all grid points
for j=1:1:Nx1
    for i=1:1:Ny1
        % Define global indexes in algebraic space
        kvx=((j-1)*Ny1+i-1)*3+1; % Vx
        kvy=kvx+1; % Vy
        kpm=kvx+2; % P
        % Reload solution
        vx(i,j)=S(kvx);
        vy(i,j)=S(kvy);
        pr(i,j)=S(kpm)*pscale;
    end
end


% Define timestep
dt=1e+30;
maxvx=max(max(abs(vx)));
maxvy=max(max(abs(vy)));
if(dt*maxvx>dxymax*dx)
    dt=dxymax*dx/maxvx;
end
if(dt*maxvy>dxymax*dy)
    dt=dxymax*dy/maxvy;
end

% Compute velocity in basic nodes
% vx, vy
for j=1:1:Nx
    for i=1:1:Ny
        vxb(i,j)=(vx(i,j)+vx(i+1,j))/2;
        vyb(i,j)=(vy(i,j)+vy(i,j+1))/2;
    end
end

% Compute velocity in internal pressure nodes
% vx
for j=2:1:Nx
    for i=2:1:Ny
        vxp(i,j)=(vx(i,j)+vx(i,j-1))/2;
    end
end
% Apply BC
% Top
vxp(1,2:Nx-1)=-bctop*vxp(2,2:Nx-1);    
% Bottom
vxp(Ny1,2:Nx-1)=-bcbottom*vxp(Ny,2:Nx-1);    
% Left
vxp(:,1)=-vxp(:,2);
% Right
vxp(:,Nx1)=-vxp(:,Nx);
% vy
for j=2:1:Nx
    for i=2:1:Ny
        vyp(i,j)=(vy(i,j)+vy(i-1,j))/2;
    end
end    
% Apply BC
% Left
vyp(2:Ny-1,1)=-bcleft*vyp(2:Ny-1,2);    
% Right
vyp(2:Ny-1,Nx1)=-bcright*vyp(2:Ny-1,Nx); % Free slip    
% Top
vyp(1,:)=-vyp(2,:);
% Bottom
vyp(Ny1,:)=-vyp(Ny,:);



figure(1);colormap('Jet');clf
subplot(2,2,1)
pcolor(x,y,RHO)
shading flat;
axis ij image;
colorbar
title('colormap of RHO')
hold on
quiver(xp(3:5:Nx1),yp(3:5:Ny1),vxp(3:5:Ny,3:5:Nx1),vyp(3:5:Ny1,3:5:Nx1),'w')

subplot(2,2,2)
pcolor(xp,yp,pr)
shading interp;
axis ij image;
colorbar
title('colormap of Pressure')
hold on
quiver(xp(3:5:Nx1),yp(3:5:Ny1),vxp(3:5:Ny,3:5:Nx1),vyp(3:5:Ny1,3:5:Nx1),'k')

subplot(2,2,3)
pcolor(xp,yp,vxp)
shading interp;
axis ij image;
colorbar
title('colormap of vx')
hold on
quiver(xp(3:5:Nx1),yp(3:5:Ny1),vxp(3:5:Ny,3:5:Nx1),vyp(3:5:Ny1,3:5:Nx1),'k')

subplot(2,2,4)
pcolor(xp,yp,vyp)
shading interp;
axis ij image;
colorbar
title('colormap of vy')
hold on
quiver(xp(3:5:Nx1),yp(3:5:Ny1),vxp(3:5:Ny,3:5:Nx1),vyp(3:5:Ny1,3:5:Nx1),'k')

xm1=xm;ym1=ym;marknum1(1)=marknum;
xm2=xm;ym2=ym;marknum2(1)=marknum;
xm3=xm;ym3=ym;marknum3(1)=marknum;
xm4=xm;ym4=ym;marknum4(1)=marknum;
timestep=0;
for timestep=timestep+1:1:ntimesteps

% LINEAR
xm=xm1;ym=ym1;marknum=marknum1(timestep);
% Move markers with 4th order Runge-Kutta
vxm=zeros(4,1);
vym=zeros(4,1);
for m=1:1:marknum
    % Save initial marker coordinates
    xA=xm(m);
    yA=ym(m);
    for rk=1:1:4
        % Interpolate vx
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xvx(1))/dx)+1;
        i=fix((ym(m)-yvx(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx-1)
            j=Nx-1;
        end
        if(i<1)
            i=1;
        elseif(i>Ny)
            i=Ny;
        end
        % Compute distances
        dxmj=xm(m)-xvx(j);
        dymi=ym(m)-yvx(i);
        % Compute weights
        % Compute vx velocity for the top and bottom of the cell
        vxm13=vx(i,j)*(1-dxmj/dx)+vx(i,j+1)*dxmj/dx;
        vxm24=vx(i+1,j)*(1-dxmj/dx)+vx(i+1,j+1)*dxmj/dx;
        % Compute vx
        vxm(rk)=(1-dymi/dy)*vxm13+(dymi/dy)*vxm24;
        
        % Interpolate vy
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xvy(1))/dx)+1;
        i=fix((ym(m)-yvy(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx)
            j=Nx;
        end
        if(i<1)
            i=1;
        elseif(i>Ny-1)
            i=Ny-1;
        end
        % Compute distances
        dxmj=xm(m)-xvy(j);
        dymi=ym(m)-yvy(i);
        % Compute weights
        % Compute vy velocity for the left and right of the cell
        vym12=vy(i,j)*(1-dymi/dy)+vy(i+1,j)*dymi/dy;
        vym34=vy(i,j+1)*(1-dymi/dy)+vy(i+1,j+1)*dymi/dy;
        % Compute vy
        vym(rk)=(1-dxmj/dx)*vym12+(dxmj/dx)*vym34;
        
        % Change coordinates to obtain B,C,D points
        if(rk==1 || rk==2)
            xm(m)=xA+dt/2*vxm(rk);
            ym(m)=yA+dt/2*vym(rk);
        elseif(rk==3)
            xm(m)=xA+dt*vxm(rk);
            ym(m)=yA+dt*vym(rk);
        end
    end
    % Restore initial coordinates
    xm(m)=xA;
    ym(m)=yA;
    % Compute effective velocity
    vxmeff=1/6*(vxm(1)+2*vxm(2)+2*vxm(3)+vxm(4));
    vymeff=1/6*(vym(1)+2*vym(2)+2*vym(3)+vym(4));
    % Move markers
    xm(m)=xm(m)+dt*vxmeff;
    ym(m)=ym(m)+dt*vymeff;
end  
xm1=xm;ym1=ym;
  

% JENNY (Jenny, 2001)
xm=xm2;ym=ym2;marknum=marknum2(timestep);
% Move markers with 4th order Runge-Kutta
vxm=zeros(4,1);
vym=zeros(4,1);
for m=1:1:marknum
    % Save initial marker coordinates
    xA=xm(m);
    yA=ym(m);
    for rk=1:1:4
        % Interpolate vxb,vyb
        % (i,j)a------(i,j+1)b
        %   |      dym=x2 |
        %   | dxm=x1 o    |
        %   |             |
        % (i+1,j)c------(i+1,j+1)d
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-x(1))/dx)+1;
        i=fix((ym(m)-y(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx-1)
            j=Nx-1;
        end
        if(i<1)
            i=1;
        elseif(i>Ny-1)
            i=Ny-1;
        end
        % Compute distances
        dxmj=xm(m)-x(j);
        dymi=ym(m)-y(i);
        % Compute weights
        wtmij=(1-dxmj/dx)*(1-dymi/dy);
        wtmi1j=(1-dxmj/dx)*(dymi/dy);    
        wtmij1=(dxmj/dx)*(1-dymi/dy);
        wtmi1j1=(dxmj/dx)*(dymi/dy);
        % Compute vx, vy velocity
        vxm(rk)=vxb(i,j)*wtmij+vxb(i+1,j)*wtmi1j+...
            vxb(i,j+1)*wtmij1+vxb(i+1,j+1)*wtmi1j1;
        vym(rk)=vyb(i,j)*wtmij+vyb(i+1,j)*wtmi1j+...
            vyb(i,j+1)*wtmij1+vyb(i+1,j+1)*wtmi1j1;
        % Compute coefficients Ue1, Ug2
        Ue1=-dx/dy*(vyb(i,j)-vyb(i,j+1)-vyb(i+1,j)+vyb(i+1,j+1));
        Ug2=-dy/dx*(vxb(i,j)-vxb(i,j+1)-vxb(i+1,j)+vxb(i+1,j+1));
        % Compute corrections dU1, dU2
        dU1=-1/2*dxmj/dx*(1-dxmj/dx)*Ue1;
        dU2=-1/2*dymi/dy*(1-dymi/dy)*Ug2;
        % Add corrections
        vxm(rk)=vxm(rk)+dU1;
        vym(rk)=vym(rk)+dU2;
        
        % Change coordinates to obtain B,C,D points
        if(rk==1 || rk==2)
            xm(m)=xA+dt/2*vxm(rk);
            ym(m)=yA+dt/2*vym(rk);
        elseif(rk==3)
            xm(m)=xA+dt*vxm(rk);
            ym(m)=yA+dt*vym(rk);
        end
    end
    % Restore initial coordinates
    xm(m)=xA;
    ym(m)=yA;
    % Compute effective velocity
    vxmeff=1/6*(vxm(1)+2*vxm(2)+2*vxm(3)+vxm(4));
    vymeff=1/6*(vym(1)+2*vym(2)+2*vym(3)+vym(4));
    % Move markers
    xm(m)=xm(m)+dt*vxmeff;
    ym(m)=ym(m)+dt*vymeff;
end  
xm2=xm;ym2=ym;

% FINAL (Gerya, 2019)
xm=xm3;ym=ym3;marknum=marknum3(timestep);
vpratio=1/3; % Weight of averaged velocity for moving markers
% Move markers with 4th order Runge-Kutta
vxm=zeros(4,1);
vym=zeros(4,1);
for m=1:1:marknum
    % Save initial marker coordinates
    xA=xm(m);
    yA=ym(m);
    for rk=1:1:4
        % Interpolate vxp,vyp
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xp(1))/dx)+1;
        i=fix((ym(m)-yp(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx)
            j=Nx;
        end
        if(i<1)
            i=1;
        elseif(i>Ny)
            i=Ny;
        end
        % Compute distances
        dxmj=xm(m)-xp(j);
        dymi=ym(m)-yp(i);
        % Compute weights
        wtmij=(1-dxmj/dx)*(1-dymi/dy);
        wtmi1j=(1-dxmj/dx)*(dymi/dy);    
        wtmij1=(dxmj/dx)*(1-dymi/dy);
        wtmi1j1=(dxmj/dx)*(dymi/dy);
        % Compute vx, vy velocity
        vxm(rk)=vxp(i,j)*wtmij+vxp(i+1,j)*wtmi1j+...
            vxp(i,j+1)*wtmij1+vxp(i+1,j+1)*wtmi1j1;
        vym(rk)=vyp(i,j)*wtmij+vyp(i+1,j)*wtmi1j+...
            vyp(i,j+1)*wtmij1+vyp(i+1,j+1)*wtmi1j1;
        
        % Interpolate vx
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xvx(1))/dx)+1;
        i=fix((ym(m)-yvx(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx-1)
            j=Nx-1;
        end
        if(i<1)
            i=1;
        elseif(i>Ny)
            i=Ny;
        end
        % Compute distances
        dxmj=xm(m)-xvx(j);
        dymi=ym(m)-yvx(i);
        % Compute weights
        wtmij=(1-dxmj/dx)*(1-dymi/dy);
        wtmi1j=(1-dxmj/dx)*(dymi/dy);    
        wtmij1=(dxmj/dx)*(1-dymi/dy);
        wtmi1j1=(dxmj/dx)*(dymi/dy);
        % Compute vx velocity
        vxm(rk)=vpratio*vxm(rk)+(1-vpratio)*(vx(i,j)*wtmij+vx(i+1,j)*wtmi1j+...
            vx(i,j+1)*wtmij1+vx(i+1,j+1)*wtmi1j1);
        
        % Interpolate vy
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xvy(1))/dx)+1;
        i=fix((ym(m)-yvy(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx)
            j=Nx;
        end
        if(i<1)
            i=1;
        elseif(i>Ny-1)
            i=Ny-1;
        end
        % Compute distances
        dxmj=xm(m)-xvy(j);
        dymi=ym(m)-yvy(i);
        % Compute weights
        wtmij=(1-dxmj/dx)*(1-dymi/dy);
        wtmi1j=(1-dxmj/dx)*(dymi/dy);    
        wtmij1=(dxmj/dx)*(1-dymi/dy);
        wtmi1j1=(dxmj/dx)*(dymi/dy);
        % Compute vx velocity
        vym(rk)=vpratio*vym(rk)+(1-vpratio)*(vy(i,j)*wtmij+vy(i+1,j)*wtmi1j+...
            vy(i,j+1)*wtmij1+vy(i+1,j+1)*wtmi1j1);        
        
        % Change coordinates to obtain B,C,D points
        if(rk==1 || rk==2)
            xm(m)=xA+dt/2*vxm(rk);
            ym(m)=yA+dt/2*vym(rk);
        elseif(rk==3)
            xm(m)=xA+dt*vxm(rk);
            ym(m)=yA+dt*vym(rk);
        end
    end
    % Restore initial coordinates
    xm(m)=xA;
    ym(m)=yA;
    % Compute effective velocity
    vxmeff=1/6*(vxm(1)+2*vxm(2)+2*vxm(3)+vxm(4));
    vymeff=1/6*(vym(1)+2*vym(2)+2*vym(3)+vym(4));
    % Move markers
    xm(m)=xm(m)+dt*vxmeff;
    ym(m)=ym(m)+dt*vymeff;
end  
xm3=xm;ym3=ym;
    
% FINAL (Gerya, 2020)
xm=xm4;ym=ym4;marknum=marknum4(timestep);
% Move markers with 4th order Runge-Kutta
vxm=zeros(4,1);
vym=zeros(4,1);
for m=1:1:marknum
    % Save initial marker coordinates
    xA=xm(m);
    yA=ym(m);
    for rk=1:1:4
        % Interpolate vx
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xvx(1))/dx)+1;
        i=fix((ym(m)-yvx(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx-1)
            j=Nx-1;
        end
        if(i<1)
            i=1;
        elseif(i>Ny)
            i=Ny;
        end
        % Compute distances
        dxmj=xm(m)-xvx(j);
        dymi=ym(m)-yvx(i);
        % Compute weights
        % Compute vx velocity for the top and bottom of the cell
        vxm13=vx(i,j)*(1-dxmj/dx)+vx(i,j+1)*dxmj/dx;
        vxm24=vx(i+1,j)*(1-dxmj/dx)+vx(i+1,j+1)*dxmj/dx;
        % Compute correction
        if(dxmj/dx>=0.5)
            if(j<Nx-1)
                vxm13=vxm13+1/2*((dxmj/dx-0.5)^2)*(vx(i,j)-2*vx(i,j+1)+vx(i,j+2));
                vxm24=vxm24+1/2*((dxmj/dx-0.5)^2)*(vx(i+1,j)-2*vx(i+1,j+1)+vx(i+1,j+2));
            end
        else
            if(j>1)
                vxm13=vxm13+1/2*((dxmj/dx-0.5)^2)*(vx(i,j-1)-2*vx(i,j)+vx(i,j+1));
                vxm24=vxm24+1/2*((dxmj/dx-0.5)^2)*(vx(i+1,j-1)-2*vx(i+1,j)+vx(i+1,j+1));
            end
        end
        % Compute vx
        vxm(rk)=(1-dymi/dy)*vxm13+(dymi/dy)*vxm24;
        
        % Interpolate vy
        % Define i,j indexes for the upper left node
        j=fix((xm(m)-xvy(1))/dx)+1;
        i=fix((ym(m)-yvy(1))/dy)+1;
        if(j<1)
            j=1;
        elseif(j>Nx)
            j=Nx;
        end
        if(i<1)
            i=1;
        elseif(i>Ny-1)
            i=Ny-1;
        end
        % Compute distances
        dxmj=xm(m)-xvy(j);
        dymi=ym(m)-yvy(i);
        % Compute weights
        % Compute vy velocity for the left and right of the cell
        vym12=vy(i,j)*(1-dymi/dy)+vy(i+1,j)*dymi/dy;
        vym34=vy(i,j+1)*(1-dymi/dy)+vy(i+1,j+1)*dymi/dy;
        % Compute correction
        if(dymi/dy>=0.5)
            if(i<Ny-1)
                vym12=vym12+1/2*((dymi/dy-0.5)^2)*(vy(i,j)-2*vy(i+1,j)+vy(i+2,j));
                vym34=vym34+1/2*((dymi/dy-0.5)^2)*(vy(i,j+1)-2*vy(i+1,j+1)+vy(i+2,j+1));
            end      
        else
            if(i>1)
                vym12=vym12+1/2*((dymi/dy-0.5)^2)*(vy(i-1,j)-2*vy(i,j)+vy(i+1,j));
                vym34=vym34+1/2*((dymi/dy-0.5)^2)*(vy(i-1,j+1)-2*vy(i,j+1)+vy(i+1,j+1));
            end
        end
        % Compute vy
        vym(rk)=(1-dxmj/dx)*vym12+(dxmj/dx)*vym34;
        
        % Change coordinates to obtain B,C,D points
        if(rk==1 || rk==2)
            xm(m)=xA+dt/2*vxm(rk);
            ym(m)=yA+dt/2*vym(rk);
        elseif(rk==3)
            xm(m)=xA+dt*vxm(rk);
            ym(m)=yA+dt*vym(rk);
        end
    end
    % Restore initial coordinates
    xm(m)=xA;
    ym(m)=yA;
    % Compute effective velocity
    vxmeff=1/6*(vxm(1)+2*vxm(2)+2*vxm(3)+vxm(4));
    vymeff=1/6*(vym(1)+2*vym(2)+2*vym(3)+vym(4));
    % Move markers
    xm(m)=xm(m)+dt*vxmeff;
    ym(m)=ym(m)+dt*vymeff;
end  
xm4=xm;ym4=ym;


% Adding markers
for models=1:1:4
if(models==1)
    xm=xm1;ym=ym1;marknum=marknum1(timestep);
elseif(models==2)
    xm=xm2;ym=ym2;marknum=marknum2(timestep);
elseif(models==3)
    xm=xm3;ym=ym3;marknum=marknum3(timestep);
elseif(models==4)
    xm=xm4;ym=ym4;marknum=marknum4(timestep);
end    
% Add markers to empty areas
mdis=1e30*ones(Nym,Nxm);
mnum=zeros(Nym,Nxm);
mtyp=zeros(Nym,Nxm);
mpor=zeros(Nym,Nxm);
xxm=dxm/2:dxm:xsize-dxm/2;
yym=dym/2:dym:ysize-dym/2;
for m=1:1:marknum
    
    % Check markers with the nearest nodes
    % Define i,j indexes for the upper left node
    j=fix((xm(m)-xxm(1))/dxm)+1;
    i=fix((ym(m)-yym(1))/dym)+1;
    if(j<1)
        j=1;
    elseif(j>Nxm-1)
        j=Nxm-1;
    end
    if(i<1)
        i=1;
    elseif(i>Nym-1)
        i=Nym-1;
    end
    
    % Check nodes
    % i,j Node
    % Compute distance
    dxmj=xm(m)-xxm(j);
    dymi=ym(m)-yym(i);
    dismij=(dxmj^2+dymi^2)^0.5;
    if(dismij<mdis(i,j))
        mdis(i,j)=dismij;
        mnum(i,j)=m;
        mtyp(i,j)=tm(m);
    end
    % i+1,j Node
    % Compute distance
    dxmj=xm(m)-xxm(j);
    dymi=ym(m)-yym(i+1);
    dismi1j=(dxmj^2+dymi^2)^0.5;
    if(dismi1j<mdis(i+1,j))
        mdis(i+1,j)=dismi1j;
        mnum(i+1,j)=m;
        mtyp(i+1,j)=tm(m);
    end
    % i,j+1 Node
    % Compute distance
    dxmj=xm(m)-xxm(j+1);
    dymi=ym(m)-yym(i);
    dismij1=(dxmj^2+dymi^2)^0.5;
    if(dismij1<mdis(i,j+1))
        mdis(i,j+1)=dismij1;
        mnum(i,j+1)=m;
        mtyp(i,j+1)=tm(m);
    end
    % i+1,j+1 Node
    % Compute distance
    dxmj=xm(m)-xxm(j+1);
    dymi=ym(m)-yym(i+1);
    dismi1j1=(dxmj^2+dymi^2)^0.5;
    if(dismi1j1<mdis(i+1,j+1))
        mdis(i+1,j+1)=dismi1j1;
        mnum(i+1,j+1)=m;
        mtyp(i+1,j+1)=tm(m);
    end
end

dii=5*Nxmc;
djj=5*Nymc;

for j=1:1:Nxm
    for i=1:1:Nym
        if(mnum(i,j)==0)
            % Serch surrounding nodes
            for jj=j-djj:1:j+djj
                for ii=i-dii:1:i+dii
                    if(ii>=1 && ii<=Nym && jj>=1 && jj<=Nxm && mnum(ii,jj)>0)
                        % Compute distance
                        m=mnum(ii,jj);
                        dxmj=xm(m)-xxm(j);
                        dymi=ym(m)-yym(i);
                        dismij=(dxmj^2+dymi^2)^0.5;
                        if(dismij<mdis(i,j))
                            mdis(i,j)=dismij;
                            mnum(i,j)=-m;
                            mtyp(i,j)=tm(m);
                        end
                    end
                end
            end
            % Add New marker
            if(mnum(i,j)<0)
                % Add marker number
                marknum=marknum+1;
                % Assign marker coordinates
                xm(marknum)=xxm(j)+(rand-0.5)*dxm;
                ym(marknum)=yym(i)+(rand-0.5)*dym;
                % Copy marker properties
                m=-mnum(i,j);
                tm(marknum)=tm(m); % Material type
                rhom(marknum)=rhom(m); % Mantle density
                etam(marknum)=etam(m); % Mantle viscosity
            end
        end
    end
end           
if(models==1)
    xm1=xm;ym1=ym;marknum1(timestep)=marknum;marknum1(timestep+1)=marknum;
elseif(models==2)
    xm2=xm;ym2=ym;marknum2(timestep)=marknum;marknum2(timestep+1)=marknum;
elseif(models==3)
    xm3=xm;ym3=ym;marknum3(timestep)=marknum;marknum3(timestep+1)=marknum;
elseif(models==4)
    xm4=xm;ym4=ym;marknum4(timestep)=marknum;marknum4(timestep+1)=marknum;
end
end









if(timestep==1 || fix((timestep)/visstep)*visstep==timestep)

figure(2);colormap('Jet');clf
subplot(2,3,1)
hold on
plot(xm1(1:marknum1(1)),ym1(1:marknum1(1)),'. k');
% plot(xm1(tm==1),ym1(tm==1),'. k');
% plot(xm1(tm==2),ym1(tm==2),'. r');
axis ij image
axis ([xsize/4 3*xsize/4 ysize/4 3*ysize/4])
hold off
title(['LINEAR timestep=',num2str(timestep)])
% subplot(1,3,2)
% hold on
% plot(xm2(tm==1),ym2(tm==1),'. k');
% plot(xm2(tm==2),ym2(tm==2),'. r');
% axis ij image
% hold off
% title(['JENNY timestep=',num2str(timestep)])
subplot(2,3,2)
hold on
plot(xm3(1:marknum3(1)),ym3(1:marknum3(1)),'. k');
% plot(xm3(tm==1),ym3(tm==1),'. k');
% plot(xm3(tm==2),ym3(tm==2),'. r');
axis ij image
axis ([xsize/4 3*xsize/4 ysize/4 3*ysize/4])
hold off
title(['EMPIRICAL timestep=',num2str(timestep)])
subplot(2,3,3)
hold on
plot(xm4(1:marknum4(1)),ym4(1:marknum4(1)),'. k');
% plot(xm4(tm==1),ym4(tm==1),'. k');
% plot(xm4(tm==2),ym4(tm==2),'. r');
axis ij image
axis ([xsize/4 3*xsize/4 ysize/4 3*ysize/4])
hold off
title(['THEORETICAL timestep=',num2str(timestep)])
subplot(2,3,4)
hold on
plot(xm1(1:marknum1(1)),ym1(1:marknum1(1)),'. k');
% plot(xm1(tm==1),ym1(tm==1),'. k');
% plot(xm1(tm==2),ym1(tm==2),'. r');
axis ij image
axis ([0 xsize/2 ysize/2 ysize])
hold off
title(['LINEAR timestep=',num2str(timestep)])
% subplot(1,3,2)
% hold on
% plot(xm2(tm==1),ym2(tm==1),'. k');
% plot(xm2(tm==2),ym2(tm==2),'. r');
% axis ij image
% hold off
% title(['JENNY timestep=',num2str(timestep)])
subplot(2,3,5)
hold on
plot(xm3(1:marknum3(1)),ym3(1:marknum3(1)),'. k');
% plot(xm3(tm==1),ym3(tm==1),'. k');
% plot(xm3(tm==2),ym3(tm==2),'. r');
axis ij image
axis ([0 xsize/2 ysize/2 ysize])
hold off
title(['EMPIRICAL timestep=',num2str(timestep)])
subplot(2,3,6)
hold on
plot(xm4(1:marknum4(1)),ym4(1:marknum4(1)),'. k');
% plot(xm4(tm==1),ym4(tm==1),'. k');
% plot(xm4(tm==2),ym4(tm==2),'. r');
axis ij image
axis ([0 xsize/2 ysize/2 ysize])
hold off
title(['THEORETICAL timestep=',num2str(timestep)])

figure(3);colormap('Jet');clf
subplot(2,3,1)
hold on
plot(xm1(1:marknum1(1)),ym1(1:marknum1(1)),'. k');
plot(xm1(marknum1(1):marknum1(timestep)),ym1(marknum1(1):marknum1(timestep)),'. r');
% plot(xm1(tm==1),ym1(tm==1),'. k');
% plot(xm1(tm==2),ym1(tm==2),'. r');
axis ij image
axis ([xsize/4 3*xsize/4 ysize/4 3*ysize/4])
hold off
title(['LINEAR timestep=',num2str(timestep)])
% subplot(1,3,2)
% hold on
% plot(xm2(tm==1),ym2(tm==1),'. k');
% plot(xm2(tm==2),ym2(tm==2),'. r');
% axis ij image
% hold off
% title(['JENNY timestep=',num2str(timestep)])
subplot(2,3,2)
hold on
plot(xm3(1:marknum3(1)),ym3(1:marknum3(1)),'. k');
plot(xm3(marknum3(1):marknum3(timestep)),ym3(marknum3(1):marknum3(timestep)),'. r');
% plot(xm3(tm==1),ym3(tm==1),'. k');
% plot(xm3(tm==2),ym3(tm==2),'. r');
axis ij image
axis ([xsize/4 3*xsize/4 ysize/4 3*ysize/4])
hold off
title(['EMPIRICAL timestep=',num2str(timestep)])
subplot(2,3,3)
hold on
plot(xm4(1:marknum4(1)),ym4(1:marknum4(1)),'. k');
plot(xm4(marknum4(1):marknum4(timestep)),ym4(marknum4(1):marknum4(timestep)),'. r');
% plot(xm4(tm==1),ym4(tm==1),'. k');
% plot(xm4(tm==2),ym4(tm==2),'. r');
axis ij image
axis ([xsize/4 3*xsize/4 ysize/4 3*ysize/4])
hold off
title(['THEORETICAL timestep=',num2str(timestep)])
subplot(2,3,4)
hold on
plot(xm1(1:marknum1(1)),ym1(1:marknum1(1)),'. k');
plot(xm1(marknum1(1):marknum1(timestep)),ym1(marknum1(1):marknum1(timestep)),'. r');
% plot(xm1(tm==1),ym1(tm==1),'. k');
% plot(xm1(tm==2),ym1(tm==2),'. r');
axis ij image
axis ([0 xsize/2 ysize/2 ysize])
hold off
title(['LINEAR timestep=',num2str(timestep)])
% subplot(1,3,2)
% hold on
% plot(xm2(tm==1),ym2(tm==1),'. k');
% plot(xm2(tm==2),ym2(tm==2),'. r');
% axis ij image
% hold off
% title(['JENNY timestep=',num2str(timestep)])
subplot(2,3,5)
hold on
plot(xm3(1:marknum3(1)),ym3(1:marknum3(1)),'. k');
plot(xm3(marknum3(1):marknum3(timestep)),ym3(marknum3(1):marknum3(timestep)),'. r');
% plot(xm3(tm==1),ym3(tm==1),'. k');
% plot(xm3(tm==2),ym3(tm==2),'. r');
axis ij image
axis ([0 xsize/2 ysize/2 ysize])
hold off
title(['EMPIRICAL timestep=',num2str(timestep)])
subplot(2,3,6)
hold on
plot(xm4(1:marknum4(1)),ym4(1:marknum4(1)),'. k');
plot(xm4(marknum4(1):marknum4(timestep)),ym4(marknum4(1):marknum4(timestep)),'. r');
% plot(xm4(tm==1),ym4(tm==1),'. k');
% plot(xm4(tm==2),ym4(tm==2),'. r');
axis ij image
axis ([0 xsize/2 ysize/2 ysize])
hold off
title(['THEORETICAL timestep=',num2str(timestep)])

figure(4)
hold on
plot(marknum1,'b')
plot(marknum3,'k')
plot(marknum4,'r')

pause(0.001)


end        

end

