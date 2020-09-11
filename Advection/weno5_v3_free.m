function [a,b,c,d] = weno5_v3_free(adv, dx, dy, dimx, dimy, bc, valW, valE, valS, valN, valX, BC )
% WENO
a=0; b=0; c=0; d=0;

nx   = size(adv,1)+0;
ny   = size(adv,2)+0;
nxe  = size(adv,1)+6;
nye  = size(adv,2)+6;
adve = zeros(nxe,nye);

% if bc==1 % no_flux
%     adve  = [1*adv(1,:); 1*adv(1,:); 1*adv(1,:);  adv; 1*adv(end,:); 1*adv(end,:); 1*adv(end,:) ];
%     adve  = [1*adve(:,1) 1*adve(:,1) 1*adve(:,1)  adve 1*adve(:,end) 1*adve(:,end) 1*adve(:,end)];
% elseif bc==2 % periodic
%     adve  = [1*adv(end-2,:);  1*adv(end-1,:);  1*adv(end,:);  adv;  1*adv(1,:);  1*adv(2,:);  1*adv(3,:) ];
%     adve  = [1*adve(:,end-2)  1*adve(:,end-1)  1*adve(:,end)  adve  1*adve(:,1)  1*adve(:,2)  1*adve(:,3)];
% elseif bc==3 % Stokes
%     delx1 = (adv(2,:)-adv(1,:)); delx2 = (adv(end,:)-adv(end-1,:));
%     adve  = [1*adv(1,:)-3*delx1; 1*adv(1,:)-2*delx1; 1*adv(1,:)-delx1;  adv; 1*adv(end,:)+delx2; 1*adv(end,:)+2*delx2; 1*adv(end,:)+3*delx2 ];
%     dely1 = (adve(:,2)-adve(:,1)); dely2 = (adve(:,end)-adve(:,end-1));
%     adve  = [1*adve(:,1)-3*dely1 1*adve(:,1)-2*dely1 1*adve(:,1)-dely1  adve 1*adve(:,end)+dely2 1*adve(:,end)+2*dely2 1*adve(:,end)+3*dely2];
% elseif bc==4 % Dirichlet
%     adve  = [2*valW-adv(3,:); 2*valW-adv(2,:); 2*valW-adv(1,:);  adv; 2*valE-adv(end,:); 2*valE-adv(end-1,:); 2*valE-adv(end-2,:)];
%     adve  = [2*valS-adve(:,3) 2*valS-adve(:,2) 2*valS-adve(:,1)  adve 2*valN-adve(:,end) 2*valN-adve(:,end-1) 2*valN-adve(:,end-2)];
% end
ind   = 1:size(adve,1); inxi = ind(4:end-3);
ind   = 1:size(adve,2); inyi = ind(4:end-3);
%%% x
if dimx==1
    
    % Extended values in x - NEW FOR SURFACE
    for j=3:nye-3
        for i=3:nxe-3
            
            ie = i-2; je = j-2;
            
            
            if BC(ie,je)==1 && BC(ie+1,je)==-1 %&& ie == 1
                adve(i  ,j) = 2*valW - adv(ie-1+1,je-1);
                adve(i-1,j) = 2*valW - adv(ie-1+2,je-1);
                adve(i-2,j) = 2*valW - adv(ie-1+3,je-1);
            elseif BC(ie,je)==2 && BC(ie-1,je)==-1 %&& ie==nx+2
                adve(i  ,j) = 2*valE - adv(ie-1-1,je-1);
                adve(i+1,j) = 2*valE - adv(ie-1-2,je-1);
                adve(i+2,j) = 2*valE - adv(ie-1-3,je-1);
            elseif BC(ie,je)==0 && BC(ie+1,je)==-1
                adve(i  ,j) = 2*valX - adv(ie-1+1,je-1);
                if ie<nx , adve(i-1,j) = 2*valX - adv(ie-1+2,je-1); end
                if ie<nx-1, adve(i-2,j) = 2*valX - adv(ie-1+3,je-1); end
             elseif BC(ie,je)==0 && BC(ie-1,je)==-1
                 adve(i  ,j) = 2*valX - adv(ie-1-1,je-1);
                 if ie>1  , adve(i+1,j) = 2*valX - adv(ie-1-2,je-1); end
                 if ie>2  , adve(i+2,j) = 2*valX - adv(ie-1-3,je-1); end;
            elseif BC(ie,je)==-1
                adve(i ,j ) = adv(ie-1,je-1);
            end
            
        end
    end
    
    
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
    
    % Extended values in x - NEW FOR SURFACE
    for j=3:nye-3
        for i=3:nxe-3
            
            ie = i-2; je = j-2;
               
            if BC(ie,je)==3 && BC(ie,je+1)==-1 %&& ie == 1
                adve(i,j  ) = 2*valW - adv(ie-1,je-1+1);
                adve(i,j-1) = 2*valW - adv(ie-1,je-1+2);
                adve(i,j-2) = 2*valW - adv(ie-1,je-1+3);
            elseif BC(ie,je)==4 && BC(ie,je-1)==-1 %&& ie==nx+2
                adve(i,j  ) = 2*valE - adv(ie-1,je-1-1);
                adve(i,j+1) = 2*valE - adv(ie-1,je-1-2);
                adve(i,j+2) = 2*valE - adv(ie-1,je-1-3);
            elseif BC(ie,je)==0 && BC(ie,je+1)==-1
                adve(i,j  ) = 2*valX - adv(ie-1,je-1+1);
                if je<ny  , adve(i,j-1) = 2*valX - adv(ie-1,je-1+2); end
                if je<ny-1, adve(i,j-2) = 2*valX - adv(ie-1,je-1+3); end
            elseif BC(ie,je)==0 && BC(ie,je-1)==-1
                adve(i,j  ) = 2*valX - adv(ie-1,je-1-1);
                if je>1, adve(i,j+1) = 2*valX - adv(ie-1,je-1-2); end
                if je>2, adve(i,j+2) = 2*valX - adv(ie-1,je-1-3); end
            elseif BC(ie,je)==-1
                adve(i ,j ) = adv(ie-1,je-1);
            end
            
        end
    end
    
    
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