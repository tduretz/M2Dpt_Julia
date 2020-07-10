function [a,b,c,d] = weno5_v2(adv, dx, dy, dimx, dimy, bc)
% WENO
a=0; b=0; c=0; d=0;

if bc==1 % no_flux
    adve  = [1*adv(1,:); 1*adv(1,:); 1*adv(1,:);  adv; 1*adv(end,:); 1*adv(end,:); 1*adv(end,:) ];
    adve  = [1*adve(:,1) 1*adve(:,1) 1*adve(:,1)  adve 1*adve(:,end) 1*adve(:,end) 1*adve(:,end)];
elseif bc==2 % periodic
    adve  = [1*adv(end-2,:);  1*adv(end-1,:);  1*adv(end,:);  adv;  1*adv(1,:);  1*adv(2,:);  1*adv(3,:) ];
    adve  = [1*adve(:,end-2)  1*adve(:,end-1)  1*adve(:,end)  adve  1*adve(:,1)  1*adve(:,2)  1*adve(:,3)];
elseif bc==3 % Stokes
    delx1 = (adv(2,:)-adv(1,:)); delx2 = (adv(end,:)-adv(end-1,:));
    adve  = [1*adv(1,:)-3*delx1; 1*adv(1,:)-2*delx1; 1*adv(1,:)-delx1;  adv; 1*adv(end,:)+delx2; 1*adv(end,:)+2*delx2; 1*adv(end,:)+3*delx2 ];
    dely1 = (adve(:,2)-adve(:,1)); dely2 = (adve(:,end)-adve(:,end-1));
    adve  = [1*adve(:,1)-3*dely1 1*adve(:,1)-2*dely1 1*adve(:,1)-dely1  adve 1*adve(:,end)+dely2 1*adve(:,end)+2*dely2 1*adve(:,end)+3*dely2];
end

ind   = 1:size(adve,1); inxi = ind(4:end-3);
ind   = 1:size(adve,2); inyi = ind(4:end-3);
%%% x
if dimx==1
    v1    = 1/dx*(adve(inxi-2,inyi)-adve(inxi-3,inyi));
    v2    = 1/dx*(adve(inxi-1,inyi)-adve(inxi-2,inyi));
    v3    = 1/dx*(adve(inxi  ,inyi)-adve(inxi-1,inyi));
    v4    = 1/dx*(adve(inxi+1,inyi)-adve(inxi  ,inyi));
    v5    = 1/dx*(adve(inxi+2,inyi)-adve(inxi+1,inyi));
    p1    = v1/3 - 7/6*v2 + 11/6*v3;
    p2    =-v2/6 + 5/6*v3 + v4/3;
    p3    = v3/3 + 5/6*v4 - v5/6;
    e     = (10^(-99) + 1e-6*max(max(max(max(v1.^2,v2.^2),v3.^2),v4.^2),v5.^2));
    s1    = 13/12*(v1-2*v2+v3).^2 + 1/4*(v1-4*v2+3*v3).^2;
    s2    = 13/12*(v2-2*v3+v4).^2 + 1/4*(v2-v4).^2;
    s3    = 13/12*(v3-2*v4+v5).^2 + 1/4*(3*v3-4*v4+v5).^2;
    a1    = 0.1./(s1+e).^2;
    a2    = 0.6./(s2+e).^2;
    a3    = 0.3./(s3+e).^2;
    w1    = a1./(a1+a2+a3);
    w2    = a2./(a1+a2+a3);
    w3    = a3./(a1+a2+a3);
    a     = w1.*p1 + w2.*p2 + w3.*p3; % minus x
    %
    v1    = 1/dx*(adve(inxi+3,inyi)-adve(inxi+2,inyi));
    v2    = 1/dx*(adve(inxi+2,inyi)-adve(inxi+1,inyi));
    v3    = 1/dx*(adve(inxi+1,inyi)-adve(inxi  ,inyi));
    v4    = 1/dx*(adve(inxi  ,inyi)-adve(inxi-1,inyi));
    v5    = 1/dx*(adve(inxi-1,inyi)-adve(inxi-2,inyi));
    p1    = v1/3 - 7/6*v2 + 11/6*v3;
    p2    =-v2/6 + 5/6*v3 + v4/3;
    p3    = v3/3 + 5/6*v4 - v5/6;
    e     = 1e-99 + 1e-6*max(max(max(max(v1.^2,v2.^2),v3.^2),v4.^2),v5.^2);
    s1    = 13/12*(v1-2*v2+v3).^2 + 1/4*(v1-4*v2+3*v3).^2;
    s2    = 13/12*(v2-2*v3+v4).^2 + 1/4*(v2-v4).^2;
    s3    = 13/12*(v3-2*v4+v5).^2 + 1/4*(3*v3-4*v4+v5).^2;
    a1    = 0.1./(s1+e).^2;
    a2    = 0.6./(s2+e).^2;
    a3    = 0.3./(s3+e).^2;
    w1    = a1./(a1+a2+a3);
    w2    = a2./(a1+a2+a3);
    w3    = a3./(a1+a2+a3);
    b     = w1.*p1 + w2.*p2 + w3.*p3; % plus x
end
%%% y
if dimy==1
    v1    = 1/dy*(adve(inxi,inyi-2)-adve(inxi,inyi-3));
    v2    = 1/dy*(adve(inxi,inyi-1)-adve(inxi,inyi-2));
    v3    = 1/dy*(adve(inxi,inyi  )-adve(inxi,inyi-1));
    v4    = 1/dy*(adve(inxi,inyi+1)-adve(inxi,inyi  ));
    v5    = 1/dy*(adve(inxi,inyi+2)-adve(inxi,inyi+1));
    p1    = v1/3 - 7/6*v2 + 11/6*v3;
    p2    =-v2/6 + 5/6*v3 + v4/3;
    p3    = v3/3 + 5/6*v4 - v5/6;
    e     = (10^(-99) + 1e-6*max(max(max(max(v1.^2,v2.^2),v3.^2),v4.^2),v5.^2));
    s1    = 13/12*(v1-2*v2+v3).^2 + 1/4*(v1-4*v2+3*v3).^2;
    s2    = 13/12*(v2-2*v3+v4).^2 + 1/4*(v2-v4).^2;
    s3    = 13/12*(v3-2*v4+v5).^2 + 1/4*(3*v3-4*v4+v5).^2;
    a1    = 0.1./(s1+e).^2;
    a2    = 0.6./(s2+e).^2;
    a3    = 0.3./(s3+e).^2;
    w1    = a1./(a1+a2+a3);
    w2    = a2./(a1+a2+a3);
    w3    = a3./(a1+a2+a3);
    c     = w1.*p1 + w2.*p2 + w3.*p3; % minus y
    %
    v1    = 1/dx*(adve(inxi,inyi+3)-adve(inxi,inyi+2));
    v2    = 1/dy*(adve(inxi,inyi+2)-adve(inxi,inyi+1));
    v3    = 1/dy*(adve(inxi,inyi+1)-adve(inxi,inyi  ));
    v4    = 1/dy*(adve(inxi,inyi  )-adve(inxi,inyi-1));
    v5    = 1/dy*(adve(inxi,inyi-1)-adve(inxi,inyi-2));
    p1    = v1/3 - 7/6*v2 + 11/6*v3;
    p2    =-v2/6 + 5/6*v3 + v4/3;
    p3    = v3/3 + 5/6*v4 - v5/6;
    e     = 1e-99 + 1e-6*max(max(max(max(v1.^2,v2.^2),v3.^2),v4.^2),v5.^2);
    s1    = 13/12*(v1-2*v2+v3).^2 + 1/4*(v1-4*v2+3*v3).^2;
    s2    = 13/12*(v2-2*v3+v4).^2 + 1/4*(v2-v4).^2;
    s3    = 13/12*(v3-2*v4+v5).^2 + 1/4*(3*v3-4*v4+v5).^2;
    a1    = 0.1./(s1+e).^2;
    a2    = 0.6./(s2+e).^2;
    a3    = 0.3./(s3+e).^2;
    w1    = a1./(a1+a2+a3);
    w2    = a2./(a1+a2+a3);
    w3    = a3./(a1+a2+a3);
    d     = w1.*p1 + w2.*p2 + w3.*p3; % plus y
end
end