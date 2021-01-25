function main

clear
clc

data = load('data41_benchmark');
correction = 1

xmin = 0;
ymin = 0;
xmax = 1;
ymax = 1;

ncx = 3;
ncy = 3;
dx  = (xmax-xmin)/ncx;
dy  = (ymax-ymin)/ncy;

xm  = 0.5;
ym  = 0.5;

xm  = 0.45;
ym  = 0.4;

xv  = xmin:dx:xmax;
yv  = ymin:dy:ymax;
xc  = xmin+dx/2:dx:xmax-dx/2;
yc  = ymin+dy/2:dy:ymax-dy/2;
xce = xmin-dx/2:dx:xmax+dx/2;
yce = ymin-dy/2:dy:ymax+dy/2;

[xc2,   yc2] = ndgrid(xc,  yc);
[xv2,   yv2] = ndgrid(xv,  yv);
[xvx2, yvx2] = ndgrid(xv, yce);
[xvy2, yvy2] = ndgrid(xce, yv);

% Vx = -yvx2;
% Vy =  xvy2;

Vx = data.Vx(1:4,1:5);
Vy = data.Vy(1:5,1:4);

VxC = 0.5*(Vx(2:end,2:end-1) +  Vx(1:end-1,2:end-1));
VyC = 0.5*(Vy(2:end-1,1:end-1) + Vy(2:end-1,2:end));

div = diff(Vx(:,2:end-1),1,1)/dx + diff(Vy(2:end-1,:),1,2)/dy

%%%%%%%%%

Nx = ncx+1;
Ny = ncy+1;
i=fix((xm-xv(1) )/dx)+1;
j=fix((ym-yce(1))/dy)+1;
if(i<1)
    i=1;
elseif(i>Nx-1)
    i=Nx-1;
end
if(j<1)
    j=1;
elseif(j>Ny)
    j=Ny;
end

%%

Vx_loc  = [Vx(i,j); Vx(i+1,j); Vx(i,j+1); Vx(i+1,j+1)];
x_locx   = ([xvx2(i,j); xvx2(i+1,j); xvx2(i,j+1); xvx2(i+1,j+1)] - xm);
y_locx   = ([yvx2(i,j); yvx2(i+1,j); yvx2(i,j+1); yvx2(i+1,j+1)] - ym);

%% Taras Vx
% Compute distances
dxmj=xm-xv(i);
dymi=ym-yce(j);
% Compute weights
% Compute vx velocity for the top and bottom of the cell
vxm13=Vx(i,j)*(1-dxmj/dx)+Vx(i+1,j)*dxmj/dx;
vxm24=Vx(i,j+1)*(1-dxmj/dx)+Vx(i+1,j+1)*dxmj/dx;
% Compute correction
if(dxmj/dx>=0.5)
    if(i<Nx-1)
        vxm13=vxm13+1/2*((dxmj/dx-0.5)^2)*(Vx(i,j)-2*Vx(i+1,j)+Vx(i+2,j));
        vxm24=vxm24+1/2*((dxmj/dx-0.5)^2)*(Vx(i,j+1)-2*Vx(i+1,j+1)+Vx(i+2,j+1));
    end
else
    if(i>1)
        vxm13=vxm13+1/2*((dxmj/dx-0.5)^2)*(Vx(i-1,j)-2*Vx(i,j)+Vx(i+1,j));
        vxm24=vxm24+1/2*((dxmj/dx-0.5)^2)*(Vx(i-1,j+1)-2*Vx(i,j+1)+Vx(i+1,j+1));
    end
end
% Compute vx
vxm=(1-dymi/dy)*vxm13+(dymi/dy)*vxm24;

%% Taras Vx

n     = length(x_locx);      % number of nodes involved in the support

% Create P matrix (as in Luos's paper)
P = [ ];
for i=1:n
    p = basis_2D(x_locx(i), y_locx(i));
    P = [ P; p' ];
end

%% Weighted least squares (see weighting in Dolbow's codes)
% r is normalised to support size
dmx  = 2*dx;
dmy  = 2*dy;             % here support size is different in y
rx   =  abs(x_locx)./dmx;
drdx = sign(x_locx)./dmx;
ry   =  abs(y_locx)./dmy;
drdy = sign(y_locx)./dmy;
% Weights and weights derivatives
wx     = zeros(size(x_locx));
wy     = zeros(size(x_locx));
dwdx   = zeros(size(x_locx));
dwdy   = zeros(size(x_locx));
% For first distance range (r>0.5 && r<=1.0)
wx1    = (4/3)-4*rx+4.*rx.*rx -(4/3)*rx.^3;
dwdx1  = (-4 + 8*rx-4*rx.^2).*drdx;
wx2    = (2/3) - 4*rx.*rx + 4*rx.^3;
dwdx2  = (-8*rx + 12*rx.^2).*drdx;
% For second distance range (r<=0.5)
wy1    = (4/3)-4*ry+4.*ry.*ry -(4/3)*ry.^3;
dwdy1  = (-4 + 8*ry-4*ry.^2).*drdy;
wy2    = (2/3) - 4*ry.*ry + 4*ry.^3;
dwdy2  = (-8*ry + 12*ry.^2).*drdy;
% Combine
wx(rx<=1.0)   = wx1(rx<=1.0);
wx(rx<=0.5)   = wx2(rx<=0.5);
wy(ry<=1.0)   = wy1(ry<=1.0);
wy(ry<=0.5)   = wy2(ry<=0.5);
dwdx(rx<=1.0) = dwdx1(rx<=1.0);
dwdx(rx<=0.5) = dwdx2(rx<=0.5);
dwdy(ry<=1.0) = dwdy1(ry<=1.0);
dwdy(ry<=0.5) = dwdy2(ry<=0.5);
w             = wx.*wy;
dwdx          = wy.*dwdx;
dwdy          = wx.*dwdy;

% Weight matrix
W    = diag(w);
dWdx = diag(dwdx);
dWdy = diag(dwdy);

% Some definitions
A    = P'*W*P;
B    = P'*W;
Ainv = inv(A);

% Compute derivative at local coordinate of vertical midface
px            = basis_2D(0, 0);     % basis
[dpxdx,dpxdy] = basis_der_2D(0, 0); % basis derivative
phix          = px'*(Ainv*B);       % basis function

% Compute derivatives of the shape functions (see Luo's paper)
dAinvdx = - Ainv*P'*dWdx*P*Ainv;
dAinvdy = - Ainv*P'*dWdy*P*Ainv;
dBdx    = P'*dWdx;
dBdy    = P'*dWdy;
dphidx  = dpxdx'*(Ainv*B) + px'*(dAinvdx*B) + px'*(Ainv*dBdx);   % basis function derivative
dphidy  = dpxdy'*(Ainv*B) + px'*(dAinvdy*B) + px'*(Ainv*dBdy);   % basis function derivative

Vxi         = sum(  phix.*Vx_loc',2);
dvxdx_lsq   = sum(dphidx.*Vx_loc',2); % negative sign for the flux (convention)
dvxdy_lsq   = sum(dphidy.*Vx_loc',2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define i,j indexes for the upper left node
i=fix((xm-xce(1))/dx)+1;
j=fix((ym-yv(1))/dy)+1;
if(i<1)
    i=1;
elseif(i>Nx)
    i=Nx;
end
if(j<1)
    j=1;
elseif(i>Ny-1)
    j=Ny-1;
end

Vy_loc    = [Vy(i,j); Vy(i+1,j); Vy(i,j+1); Vy(i+1,j+1)];
x_locy     = ([xvy2(i,j); xvy2(i+1,j); xvy2(i,j+1); xvy2(i+1,j+1)] - xm);
y_locy     = ([yvy2(i,j); yvy2(i+1,j); yvy2(i,j+1); yvy2(i+1,j+1)] - ym);

%% Taras Vy

% Compute distances
dxmj=xm-xce(i);
dymi=ym-yv(j);
% Compute weights
% Compute vy velocity for the left and right of the cell
vym12=Vy(i,j)*(1-dymi/dy)+Vy(i,j+1)*dymi/dy;
vym34=Vy(i+1,j)*(1-dymi/dy)+Vy(i+1,j+1)*dymi/dy;
% Compute correction
if correction == 1
if(dymi/dy>=0.5)
    if(j<Ny-1)
        vym12=vym12+1/2*((dymi/dy-0.5)^2)*(Vy(i,j)-2*Vy(i,j+1)+Vy(i,j+1));
        vym34=vym34+1/2*((dymi/dy-0.5)^2)*(Vy(i+1,j)-2*Vy(i+1,j+1)+Vy(i+1,j+2));
    end
else
    if(j>1)
        vym12=vym12+1/2*((dymi/dy-0.5)^2)*(Vy(i,j-1)-2*Vy(i,j)+Vy(i,j+1));
        vym34=vym34+1/2*((dymi/dy-0.5)^2)*(Vy(i+1,j-1)-2*Vy(i+1,j)+Vy(i+1,j+1));
    end
end
end
% Compute vy
vym=(1-dxmj/dx)*vym12+(dxmj/dx)*vym34;

%%%%% Taras

%% LSQ for y -component

n     = length(x_locy);      % number of nodes involved in the support

% Create P matrix (as in Luos's paper)
P = [ ];
for i=1:n
    p = basis_2D(x_locy(i), y_locy(i));
    P = [ P; p' ];
end

%% Weighted least squares (see weighting in Dolbow's codes)
% r is normalised to support size
dmx  = 2*dx;
dmy  = 2*dy;             % here support size is different in y
rx   =  abs(x_locy)./dmx;
drdx = sign(x_locy)./dmx;
ry   =  abs(y_locy)./dmy;
drdy = sign(y_locy)./dmy;
% Weights and weights derivatives
wx     = ones(size(x_locy));
wy     = ones(size(x_locy));
dwdx   = zeros(size(x_locy));
dwdy   = zeros(size(x_locy));
% For first distance range (r>0.5 && r<=1.0)
wx1    = (4/3)-4*rx+4.*rx.*rx -(4/3)*rx.^3;
dwdx1  = (-4 + 8*rx-4*rx.^2).*drdx;
wx2    = (2/3) - 4*rx.*rx + 4*rx.^3;
dwdx2  = (-8*rx + 12*rx.^2).*drdx;
% For second distance range (r<=0.5)
wy1    = (4/3)-4*ry+4.*ry.*ry -(4/3)*ry.^3;
dwdy1  = (-4 + 8*ry-4*ry.^2).*drdy;
wy2    = (2/3) - 4*ry.*ry + 4*ry.^3;
dwdy2  = (-8*ry + 12*ry.^2).*drdy;
% Combine
wx(rx<=1.0)   = wx1(rx<=1.0);
wx(rx<=0.5)   = wx2(rx<=0.5);
wy(ry<=1.0)   = wy1(ry<=1.0);
wy(ry<=0.5)   = wy2(ry<=0.5);
dwdx(rx<=1.0) = dwdx1(rx<=1.0);
dwdx(rx<=0.5) = dwdx2(rx<=0.5);
dwdy(ry<=1.0) = dwdy1(ry<=1.0);
dwdy(ry<=0.5) = dwdy2(ry<=0.5);
w             = wx.*wy;
dwdx          = wy.*dwdx;
dwdy          = wx.*dwdy;

% Weight matrix
W    = diag(w);
dWdx = diag(dwdx);
dWdy = diag(dwdy);

% Some definitions
A    = P'*W*P;
B    = P'*W;
Ainv = inv(A);

% Compute derivative at local coordinate of vertical midface
px            = basis_2D(0, 0);     % basis
[dpxdx,dpxdy] = basis_der_2D(0, 0); % basis derivative
phix          = px'*(Ainv*B);       % basis function

% Compute derivatives of the shape functions (see Luo's paper)
dAinvdx = - Ainv*P'*dWdx*P*Ainv;
dAinvdy = - Ainv*P'*dWdy*P*Ainv;
dBdx    = P'*dWdx;
dBdy    = P'*dWdy;
dphidx  = dpxdx'*(Ainv*B) + px'*(dAinvdx*B) + px'*(Ainv*dBdx);   % basis function derivative
dphidy  = dpxdy'*(Ainv*B) + px'*(dAinvdy*B) + px'*(Ainv*dBdy);   % basis function derivative

Vyi         = sum(  phix.*Vy_loc',2);
dvydx_lsq   = sum(dphidx.*Vy_loc',2); % negative sign for the flux (convention)
dvydy_lsq   = sum(dphidy.*Vy_loc',2);

divm =dvxdx_lsq + dvydy_lsq;

fprintf('\nResults for Taras scheme\n')
fprintf('Vx     = %2.8e\n', vxm  );
fprintf('Vy     = %2.8e\n', vym  );
% fprintf('div(V) = %2.8e\n', divm );

fprintf('\nResults for classical LSQ interp\n')
fprintf('Vx     = %2.8e\n', Vxi  );
fprintf('Vy     = %2.8e\n', Vyi  );
fprintf('div(V) = %2.8e\n', divm );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% WORKING FROM CENTROIDS FOR RBF INTERPOLATIONS

% Define i,j indexes for the upper left node
i=fix((xm-xc(1))/dx)+1;
j=fix((ym-yc(1))/dy)+1;
if(i<1)
    i=1;
elseif(i>Nx-2)
    i=Nx-2;
end
if(j<1)
    j=1;
elseif(i>Ny-2)
    j=Ny-2;
end

Vxp    = [VxC(i,j); VxC(i+1,j); VxC(i,j+1); VxC(i+1,j+1)];
Vyp    = [VyC(i,j); VyC(i+1,j); VyC(i,j+1); VyC(i+1,j+1)];
x_locp = [xc2(i,j); xc2(i+1,j); xc2(i,j+1); xc2(i+1,j+1)];
y_locp = [yc2(i,j); yc2(i+1,j); yc2(i,j+1); yc2(i+1,j+1)];

x_ij   = zeros(4,4);
y_ij   = zeros(4,4);
r_ij   = zeros(4,4);

epsi  = (1/8)^2;

for j=1:4
    for i=1:4
       xx  = abs(x_locp(i) - x_locp(j));
       yy  = abs(y_locp(i) - y_locp(j));
       x_ij(i,j) = xx;
       y_ij(i,j) = yy;
       r_ij(i,j) = sqrt(xx^2 + yy^2);
    end
end

%%%%%%%

%% Regular RBF for V
rp2    = r_ij.^2;
PHI    = exp(-epsi*rp2);
cx     = PHI \ Vxp;
cy     = PHI \ Vyp;

X      = abs(x_locp-xm);
Y      = abs(y_locp-ym);
rp2    = X.^2 + Y.^2;
phi    = exp(-epsi.*rp2);
dphidx = -2*epsi*X.*exp(-epsi.*rp2);
dphidy = -2*epsi*Y.*exp(-epsi.*rp2);
 
vxrbf = sum(   phi.*cx);
vyrbf = sum(   phi.*cy);
dvxdx = sum(dphidx.*cx);
dvydy = sum(dphidy.*cy);
div   = dvxdx + dvydy;

fprintf('\nResults for classical RBF interp\n')
fprintf('Vx     = %2.8e\n', vxrbf);
fprintf('Vy     = %2.8e\n', vyrbf);
fprintf('div(V) = %2.8e\n', div  );

%% Divergence free RBF
% From centroids

rp2   = r_ij.^2;
phi11 = -2*epsi*(2*epsi*y_ij.^2 - 1) .* exp(-epsi.*rp2);
phi12 =  4*epsi^2*x_ij.*y_ij.*exp(-epsi.*rp2);
phi21 = phi12;
phi22 = -2*epsi*(2*epsi*x_ij.^2 - 1) .* exp(-epsi.*rp2);

A      = [(phi11) (phi12); (phi21) (phi22);];
d      = [Vxp; Vyp];% + [0.5*ones(size(Vxp)); 0.5*ones(size(Vyp))];

c      = A\d;
cx     = c(1:4);
cy     = c(5:end);

X      = abs(x_locp-xm);
Y      = abs(y_locp-ym);
rp2    = X.^2 + Y.^2;
phi11 = -2*epsi*(2*epsi*Y.^2 - 1) .* exp(-epsi.*rp2);
phi12 =  4*epsi^2*X.*Y.*exp(-epsi*rp2);
phi21 = phi12;
phi22 = -2*epsi*(2*epsi*X.^2 - 1) .* exp(-epsi.*rp2);

dphi11dx = 4*epsi.^2.*X.*(2*Y.^2.*epsi - 1).*exp(-epsi.*rp2);
dphi21dx = -8*X.^2.*Y.*epsi.^3.*exp(-epsi.*rp2) + 4*Y.*epsi.^2.*exp(-epsi.*rp2);
dphi12dy = -8*X.*Y.^2.*epsi.^3.*exp(-epsi.*rp2) + 4*X.*epsi.^2.*exp(-epsi.*rp2);
dphi22dy = 4*Y.*epsi.^2.*(2*X.^2.*epsi - 1).*exp(-epsi.*rp2);

vxrbf  = sum(phi11.*cx) + sum(phi21.*cy);
vyrbf  = sum(phi12.*cx) + sum(phi22.*cy);
dvxdx  = sum(dphi11dx.*cx) + sum(dphi21dx.*cy);
dvydy  = sum(dphi12dy.*cx) + sum(dphi22dy.*cy);
div    = dvxdx + dvydy;

fprintf('\nResults for divergence-free RBF interp\n')
fprintf('Vx     = %2.8e\n', vxrbf);
fprintf('Vy     = %2.8e\n', vyrbf);
fprintf('div(V) = %2.8e\n', div  );


%%

Nx = ncx+1;
Ny = ncy+1;
i=fix((xm-xv(1) )/dx)+1;
j=fix((ym-yce(1))/dy)+1;
if(i<1)
    i=1;
elseif(i>Nx-1)
    i=Nx-1;
end
if(j<1)
    j=1;
elseif(j>Ny)
    j=Ny;
end

Vxx  = [Vx(i,j); Vx(i+1,j); Vx(i,j+1); Vx(i+1,j+1)];
xlocx   = ([xvx2(i,j); xvx2(i+1,j); xvx2(i,j+1); xvx2(i+1,j+1)]);
ylocx   = ([yvx2(i,j); yvx2(i+1,j); yvx2(i,j+1); yvx2(i+1,j+1)]);

% Define i,j indexes for the upper left node
i=fix((xm-xce(1))/dx)+1;
j=fix((ym-yv(1))/dy)+1;
if(i<1)
    i=1;
elseif(i>Nx)
    i=Nx;
end
if(j<1)
    j=1;
elseif(i>Ny-1)
    j=Ny-1;
end

Vyy    = [Vy(i,j); Vy(i+1,j); Vy(i,j+1); Vy(i+1,j+1)];
xlocy     = ([xvy2(i,j); xvy2(i+1,j); xvy2(i,j+1); xvy2(i+1,j+1)]);
ylocy     = ([yvy2(i,j); yvy2(i+1,j); yvy2(i,j+1); yvy2(i+1,j+1)]);

% Vxx = Vxp;
% Vyy = Vyp;
% xlocx = x_locp;
% ylocx = y_locp;
% xlocy = x_locp;
% ylocy = y_locp;

x11   = zeros(4,4);
x12   = zeros(4,4);
y11   = zeros(4,4);
y12   = zeros(4,4);
r11   = zeros(4,4);
r12   = zeros(4,4);
x21   = zeros(4,4);
x22   = zeros(4,4);
y21   = zeros(4,4);
y22   = zeros(4,4);
r21   = zeros(4,4);
r22   = zeros(4,4);

epsi  = (1/8)^2;

for j=1:4
    for i=1:4
       xx  = abs(xlocx(i) - xlocx(j));
       yx  = abs(ylocy(i) - ylocx(j));
       yy  = abs(ylocx(i) - ylocx(j));
       xy  = abs(xlocy(i) - xlocx(j));
       x11(i,j) = xx;
       y11(i,j) = yy;
       x12(i,j) = xy;
       y12(i,j) = yx;
       r11(i,j) = (xx^2 + yy^2);
       r12(i,j) = (xy^2 + yx^2);
    end
end

for j=1:4
    for i=1:4
       xx  = abs(xlocy(i) - xlocy(j));
       yx  = abs(ylocx(i) - ylocy(j));
       yy  = abs(ylocy(i) - ylocy(j));
       xy  = abs(xlocx(i) - xlocy(j));
       x21(i,j) = xy;
       y21(i,j) = yx;
       x22(i,j) = xx;
       y22(i,j) = yy;
       r21(i,j) = (xx^2 + yy^2);
       r22(i,j) = (xy^2 + yx^2);
    end
end

phi11 = -2*epsi*(2*epsi*y11.^2 - 1) .* exp(-epsi.*r11);
phi12 =  4*epsi^2*x21.*y12.*exp(-epsi.*r12);
phi21 =  4*epsi^2*x21.*y21.*exp(-epsi.*r21);
phi22 = -2*epsi*(2*epsi*x22.^2 - 1) .* exp(-epsi.*r22);

A      = [(phi11) (phi12); (phi21) (phi22);];
d      = [Vxx; Vyy];% + [0.5*ones(size(Vxp)); 0.5*ones(size(Vyp))];

c      = A\d;
cx     = c(1:4);
cy     = c(5:end);

x11    = abs(xlocx-xm);
y11    = abs(ylocx-ym);
x12    = abs(xlocy-xm);
y12    = abs(ylocy-ym);
x21    = x11;
y21    = y11;
x22    = x12;
y22    = y12;

r11    =  x11.^2 + y11.^2;
r12    =  x12.^2 + y12.^2;
r21    =  r11;
r22    =  r12;

phi11 = -2*epsi*(2*epsi*y11.^2 - 1) .* exp(-epsi.*r11);
phi12 =  4*epsi^2*x12.*y12.*exp(-epsi*r12);
phi21 =  4*epsi^2*x21.*y21.*exp(-epsi*r21);
phi22 = -2*epsi*(2*epsi*x22.^2 - 1) .* exp(-epsi.*r22);

dphi11dx = 4*epsi.^2.*x11.*(2*y11.^2.*epsi - 1).*exp(-epsi.*r11);
dphi21dx = -8*x12.^2.*y12.*epsi.^3.*exp(-epsi.*r12) + 4*y12.*epsi.^2.*exp(-epsi.*r12);
dphi12dy = -8*x21.*y21.^2.*epsi.^3.*exp(-epsi.*r21) + 4*x21.*epsi.^2.*exp(-epsi.*r21);
dphi22dy = 4*y22.*epsi.^2.*(2*x22.^2.*epsi - 1).*exp(-epsi.*r22);

vxrbf  = sum(phi11.*cx) + sum(phi21.*cy);
vyrbf  = sum(phi12.*cx) + sum(phi22.*cy);
dvxdx  = sum(dphi11dx.*cx) + sum(dphi21dx.*cy);
dvydy  = sum(dphi12dy.*cx) + sum(dphi22dy.*cy);
div    = dvxdx + dvydy;

fprintf('\nResults for staggered divergence-free RBF interp\n')
fprintf('Vx     = %2.8e\n', vxrbf);
fprintf('Vy     = %2.8e\n', vyrbf);
fprintf('div(V) = %2.8e\n', div  );

%%

figure(1), clf
subplot(131)
hold on
mesh(xv2, yv2, zeros(size(xv2)))
quiver(xc2, yc2, VxC, VyC)
plot(xm, ym, '*')
subplot(132)
imagesc(xv,yce,Vx'), axis xy, colorbar
subplot(133)
imagesc(xce,yv,Vy'), axis xy, colorbar

end

function b = basis_2D(xi, yi)
b  = [1; xi; yi; xi*yi;];
% b  = [1; xi; yi; xi^2; xi*yi; yi^2 ];
% b  = [1; xi; yi; xi*yi; xi^2; yi^2 ];
end

function [bx,by] = basis_der_2D(xi, yi)
bx = [0;  1;  0;     yi;  ];
by = [0;  0;  1;     xi;  ];
% bx = [0;  1;  0; 2*xi;   yi;    0 ];
% by = [0;  0;  1;    0;   xi; 2*yi ];
% % bx = [0;  1;  0; yi; 2*xi;      0 ];
% % by = [0;  0;  1; xi;   0;    2*yi ];
end


