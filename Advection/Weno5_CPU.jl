const USE_GPU  = false
const USE_MPI  = false
const DAT      = Float64   # Precision (Float64 or Float32)
include("../HT/Macros.jl")
include("./AdvectionSchemes.jl")
using Base.Threads         # Before starting Julia do 'export JULIA_NUM_THREADS=12' (used for loops with @threads macro).
using Plots

############################################################

@views function VxPlusMinus(Vxm::DatArray_k, Vxp::DatArray_k, Vx::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if ( Vx[ix+0,iy,iz] < 0.0 ) Vxm[ix,iy,iz] = Vx[ix+0,iy,iz]
        else                        Vxm[ix,iy,iz] = 0.0
        end
        if ( Vx[ix+1,iy,iz] > 0.0 ) Vxp[ix,iy,iz] = Vx[ix+1,iy,iz]
        else                        Vxp[ix,iy,iz] = 0.0
        end
        # if (@participate_a(vxm) )  @all(vxm) = @all(R) +  Dx*@d_xa(qx) + Dy*@d_ya(qy) + Dz*@d_za(qz); end
    end
    return nothing
end

@views function VyPlusMinus(Vym::DatArray_k, Vyp::DatArray_k, Vy::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if ( Vy[ix,iy+0,iz] < 0.0 ) Vym[ix,iy,iz] = Vy[ix,iy+0,iz]
        else                        Vym[ix,iy,iz] = 0.0
        end
        if ( Vy[ix,iy+1,iz] > 0.0 ) Vyp[ix,iy,iz] = Vy[ix,iy+1,iz]
        else                        Vyp[ix,iy,iz] = 0.0
        end
        # if (@participate_a(vxm) )  @all(vxm) = @all(R) +  Dx*@d_xa(qx) + Dy*@d_ya(qy) + Dz*@d_za(qz); end
    end
    return nothing
end

@views function VzPlusMinus(Vzm::DatArray_k, Vzp::DatArray_k, Vz::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if ( Vz[ix,iy,iz+0] < 0.0 ) Vzm[ix,iy,iz] = Vz[ix,iy,iz+0]
        else                        Vzm[ix,iy,iz] = 0.0
        end
        if ( Vz[ix,iy,iz+1] > 0.0 ) Vzp[ix,iy,iz] = Vz[ix,iy,iz+1]
        else                        Vzp[ix,iy,iz] = 0.0
        end
        # if (@participate_a(vxm) )  @all(vxm) = @all(R) +  Dx*@d_xa(qx) + Dy*@d_ya(qy) + Dz*@d_za(qz); end
    end
    return nothing
end


@views function CrazyMax(e::DatArray_k, v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    # e     = 10^(-99) + 1e-6*max(max(max(max(v1.^2,v2.^2),v3.^2),v4.^2),v5.^2);
    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        result = max(max(max(max(v1[ix,iy,iz]^2,v2[ix,iy,iz].^2), v3[ix,iy,iz]^2), v4[ix,iy,iz]^2), v5[ix,iy,iz]^2);
        e[ix,iy,iz]     = 10^(-99) + 1e-6*result
    end
    return nothing
end

@views function ComputeCoeffs(p1::DatArray_k, p2::DatArray_k, p3::DatArray_k, v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        if @participate_a(p1) @all(p1) = @all(v1)/3.0 - 7.0/6.0*@all(v2) + 11.0/6.0*@all(v3); end
        if @participate_a(p2) @all(p2) =-@all(v2)/6.0 + 5.0/6.0*@all(v3) + @all(v4)/3.0;      end
        if @participate_a(p3) @all(p3) = @all(v3)/3.0 + 5.0/6.0*@all(v4) - @all(v5)/6.0;      end
    end
    return nothing
end

############################################################

@views function MainWeno()

Vizu   = 1
Upwind = 0

xmin     = -0.0;  xmax    =    1; Lx = xmax - xmin;
ymin     = -0.0;  ymax    =    1; Ly = ymax - ymin;
zmin     = -0.00; zmax    =    1; Lz = zmax - zmin;

# Numerics
nt       = 1
nout     = 100;
nx       = 4*32;
ny       = 4*32;
nz       = 4*32;

Nix      = USE_MPI ? nx_g() : nx;                                               #SO: TODO: this is obtained from the global_grid for MPI.
Niy      = USE_MPI ? ny_g() : ny;                                               #SO: this is obtained from the global_grid.
Niz      = USE_MPI ? nz_g() : nz;
dx       = Lx/Nix;                                                              #SO: why not dx = Lx/(nx-1) or dx = Lx/(Nix-1) respectively
dy       = Ly/Niy;
dz       = Lz/Niz;


Vx      =  myones(nx+1,ny+0,nz+0);
Vy      =  myzeros(nx+0,ny+1,nz+0);
Vz      =  myzeros(nx+0,ny+0,nz+1);
# Vx      =  myones(nx+1,ny+0,nz+0);
# Vy      =  myones(nx+0,ny+1,nz+0);
# Vz      =  myzeros(nx+0,ny+0,nz+1);

VxC     =  myzeros(nx+0,ny+0,nz+0);
VyC     =  myzeros(nx+0,ny+0,nz+0);
VzC     =  myzeros(nx+0,ny+0,nz+0);
Tc_ex   =  myzeros(nx+2,ny+2,nz+2);

@. VxC     =  0.5*(Vx[1:end-1,:,:] + Vx[2:end-0,:,:] );
@. VyC     =  0.5*(Vy[:,1:end-1,:] + Vy[:,2:end-0,:] );
@. VzC     =  0.5*(Vz[:,:,1:end-1] + Vz[:,:,2:end-0] );

xc  = LinRange(xmin+dx/2, xmax-dx/2, nx)
yc  = LinRange(ymin+dy/2, ymax-dy/2, ny)
zc  = LinRange(xmin+dz/2, zmax-dz/2, nz)
xce = LinRange(xmin-dx/2, xmax+dx/2, nx+2)
yce = LinRange(ymin-dy/2, ymax+dy/2, ny+2)
zce = LinRange(xmin-dz/2, zmax+dz/2, nz+2)
xv  = LinRange(xmin, xmax, nx+1)
yv  = LinRange(ymin, ymax, ny+1)
zv  = LinRange(xmin, zmax, nz+1)
(xce2,yce2,zce2) = ([x for x=xce,y=yce,z=zce], [y for x=xce,y=yce,z=zce], [z for x=xce,y=yce,z=zce]); #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
(xc2,yc2,zc2)    = ([x for x=xc,y=yc,z=zc], [y for x=xc,y=yc,z=zc], [z for x=xc,y=yc,z=zc]); #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
(xv2,yv2,zv2)    = ([x for x=xv,y=yv,z=zv], [y for x=xv,y=yv,z=zv], [z for x=xv,y=yv,z=zv]);
@printf("Grid was set up!\n")

xC = 0.1
yC = 0.1
zC = 0.5*(zmin+zmax)
# xC = 0.1
# yC = 0.1
# zC = 0.5*(zmin+zmax)
@. Tc_ex = exp(-(xce2-xC)^2/ 0.001 - (yce2-yC)^2/ 0.001 - (zce2-zC)^2/ 0.001)
display(maximum_g(Tc_ex))

# Compute Courant criteria
dt = 0.25*min(dx,dy,dz) / max( maximum_g(Vx), maximum_g(Vy), maximum_g(Vz))
display(max( maximum_g(Vx), maximum_g(Vy), maximum_g(Vz)))
display(dt)

# upwind velocities
Vxm     =  myzeros(nx+0,ny+0,nz+0);
Vxp     =  myzeros(nx+0,ny+0,nz+0);
VxPlusMinus(Vxm, Vxp, Vx, nx, ny, nz)
Vym     =  myzeros(nx+0,ny+0,nz+0);
Vyp     =  myzeros(nx+0,ny+0,nz+0);
VyPlusMinus(Vym, Vyp, Vy, nx, ny, nz)
Vzm     =  myzeros(nx+0,ny+0,nz+0);
Vzp     =  myzeros(nx+0,ny+0,nz+0);
VzPlusMinus(Vzm, Vzp, Vz, nx, ny, nz)

dTdxm     =  myzeros(nx+0,ny+0,nz+0);
dTdxp     =  myzeros(nx+0,ny+0,nz+0);

# display(minimum_g(Vxm))
# display(minimum_g(Vxp))
# display(minimum_g(Vym))
# display(minimum_g(Vyp))
# display(minimum_g(Vzm))
# display(minimum_g(Vzp))

order = 2;
a     = 0;
b     = 0;
c     = 0;
d     = 0;

Tc  =  Tc_ex[2:end-1,2:end-1,2:end-1]

nit = 50;

Tc_exxx = myzeros(nx+6,ny+6,nz+6)
v1  =  myzeros(nx+0,ny+0,nz+0);
v2  =  myzeros(nx+0,ny+0,nz+0);
v3  =  myzeros(nx+0,ny+0,nz+0);
v4  =  myzeros(nx+0,ny+0,nz+0);
v5  =  myzeros(nx+0,ny+0,nz+0);
p1  =  myzeros(nx+0,ny+0,nz+0);
p2  =  myzeros(nx+0,ny+0,nz+0);
p3  =  myzeros(nx+0,ny+0,nz+0);
w1  =  myzeros(nx+0,ny+0,nz+0);
w2  =  myzeros(nx+0,ny+0,nz+0);
w3  =  myzeros(nx+0,ny+0,nz+0);
e   =  myzeros(nx+0,ny+0,nz+0);
dTdxp  =  myzeros(nx+0,ny+0,nz+0);
dTdxm  =  myzeros(nx+0,ny+0,nz+0);
Told   =  myzeros(nx+0,ny+0,nz+0);

time = 0


for it=1:50

    time += dt

    if Upwind == 0

    @. Told = Tc;
    for io=1:order

        # Weno 5
        @. Tc_exxx[4:end-3,4:end-3,4:end-3] = Tc;
        @. Tc_exxx[3    ,4:end-3,4:end-3] =  Tc[1  ,:,:]; @. Tc_exxx[2    ,4:end-3,4:end-3] =  Tc[1  ,:,:];; @. Tc_exxx[1    ,4:end-3,4:end-3] =  Tc[  1,:,:];
        @. Tc_exxx[end  ,4:end-3,4:end-3] =  Tc[end,:,:]; @. Tc_exxx[end-1,4:end-3,4:end-3] =  Tc[end,:,:];; @. Tc_exxx[end-2,4:end-3,4:end-3] =  Tc[end,:,:];

            @. v1    = 1.0/dx*(Tc_exxx[2:end-5,4:end-3,4:end-3]-Tc_exxx[1:end-6,4:end-3,4:end-3]);
            @. v2    = 1.0/dx*(Tc_exxx[3:end-4,4:end-3,4:end-3]-Tc_exxx[2:end-5,4:end-3,4:end-3]);
            @. v3    = 1.0/dx*(Tc_exxx[4:end-3,4:end-3,4:end-3]-Tc_exxx[3:end-4,4:end-3,4:end-3]);
            @. v4    = 1.0/dx*(Tc_exxx[5:end-2,4:end-3,4:end-3]-Tc_exxx[4:end-3,4:end-3,4:end-3]);
            @. v5    = 1.0/dx*(Tc_exxx[7:end-0,4:end-3,4:end-3]-Tc_exxx[5:end-2,4:end-3,4:end-3]);
            # @kernel cublocks cuthreads ComputeCoeffs(p1, p2, p3, v1, v2, v3, v4, v5, nx, ny, nz); @devicesync();
            # @. p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3;
            # @. p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0;
            # @. p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0;
            @kernel cublocks cuthreads CrazyMax(e, v1, v2, v3, v4, v5, nx, ny, nz);               @devicesync();
            @. w1    = 13.0/12.0*(v1-2.0*v2+v3)^2.0 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2.0;
            @. w2    = 13.0/12.0*(v2-2.0*v3+v4)^2.0 + 1.0/4.0*(v2-v4)^2.0;
            @. w3    = 13.0/12.0*(v3-2.0*v4+v5)^2.0 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2.0;
            @. w1    = 0.1/(w1+e)^2.0;
            @. w2    = 0.6/(w2+e)^2.0;
            @. w3    = 0.3/(w3+e)^2.0;
            @. e     = (w1+w2+w3)
            @. w1    = w1/e;
            @. w2    = w2/e;
            @. w3    = w3/e;
            @. dTdxm = w1*p1 + w2*p2 + w3*p3; # minus x
            #
            # @. v1    = 1.0/dx*(Tc_exxx[7:end+0,4:end-3,4:end-3]-Tc_exxx[6:end-1,4:end-3,4:end-3]);
            # @. v2    = 1.0/dx*(Tc_exxx[6:end-1,4:end-3,4:end-3]-Tc_exxx[5:end-2,4:end-3,4:end-3]);
            # @. v3    = 1.0/dx*(Tc_exxx[5:end-2,4:end-3,4:end-3]-Tc_exxx[4:end-3,4:end-3,4:end-3]);
            # @. v4    = 1.0/dx*(Tc_exxx[4:end-3,4:end-3,4:end-3]-Tc_exxx[3:end-4,4:end-3,4:end-3]);
            # @. v5    = 1.0/dx*(Tc_exxx[3:end-4,4:end-3,4:end-3]-Tc_exxx[2:end-5,4:end-3,4:end-3]);
            # @. p1    = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3;
            # @. p2    =-v2/6.0 + 5.0/6.0*v3 + v4/3.0;
            # @. p3    = v3/3.0 + 5.0/6.0*v4 - v5/6.0;
            # CrazyMax(e, v1, v2, v3, v4, v5, nx, ny, nz)
            # @. w1    = 13.0/12.0*(v1-2.0*v2+v3)^2.0 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2.0;
            # @. w2    = 13.0/12.0*(v2-2.0*v3+v4)^2.0 + 1.0/4.0*(v2-v4)^2.0;
            # @. w3    = 13.0/12.0*(v3-2.0*v4+v5)^2.0 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2.0;
            # @. w1    = 0.1/(w1+e)^2.0;
            # @. w2    = 0.6/(w2+e)^2.0;
            # @. w3    = 0.3/(w3+e)^2.0;
            # @. e     = (w1+w2+w3)
            # @. w1    = w1/e;
            # @. w2    = w2/e;
            # @. w3    = w3/e;
            # @. dTdxp = w1*p1 + w2*p2 + w3*p3; # minus x

        @. Tc = Tc - dt*(Vxp*dTdxm + Vxm*dTdxp);
    end
    @. Tc = (1.0/order)*Tc + (1.0-1.0/order)*Told;

    end

    if Upwind == 1
    # # Function call - super slow
    # Upwind(Tc,Told,Tc_ex,dTdxp,dTdxm,Vxp,Vxm,Vyp,Vym,Vzp,Vzm,dx,dy,dz,dt,order)

    # # Without function call - very slow
    @. Told = Tc;
      for io=1:order
          @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
          @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
          @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
          @. dTdxp = 1.0/dx*(Tc_ex[3:end-0,2:end-1,2:end-1] - Tc_ex[2:end-1,2:end-1,2:end-1])
          @. dTdxm = 1.0/dx*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[1:end-2,2:end-1,2:end-1])
          @. Tc    = Tc - dt*(Vxp*dTdxm + Vxm*dTdxp);
      end
      @. Tc = (1.0/order)*Tc + (1.0-(1.0/order))*Told;

      @. Told = Tc
      for io=1:order
          @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
          @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
          @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
          @. dTdxp = 1.0/dy*(Tc_ex[2:end-1,3:end-0,2:end-1] - Tc_ex[2:end-1,2:end-1,2:end-1])
          @. dTdxm = 1.0/dy*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[2:end-1,1:end-2,2:end-1])
          @. Tc    = Tc - dt*(Vyp*dTdxm + Vym*dTdxp);
      end
      @. Tc = (1.0/order)*Tc + (1.0-(1.0/order))*Told;

      @. Told = Tc;
      for io=1:order
          @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
          @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
          @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
          @. dTdxp = 1.0/dz*(Tc_ex[2:end-1,2:end-1,3:end-0] - Tc_ex[2:end-1,2:end-1,2:end-1])
          @. dTdxm = 1.0/dz*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[2:end-1,2:end-1,1:end-2])
          @. Tc    = Tc - dt*(Vzp*dTdxm + Vzm*dTdxp);
      end
      @. Tc = (1.0/order)*Tc + (1.0-1.0/order)*Told;

  end

end

display(maximum_g(Tc_ex[2:end-1,2:end-1,2:end-1]))
display(xC + time*1.0)
display(yC + time*1.0)

if (Vizu == 1)
    X = Tc
    display( heatmap(xc, yc, transpose(X[:,:,Int(ceil(nz/2))]),c=:viridis,aspect_ratio=1) );
    # fx = VxC[:,:,Int(ceil(nz/2))]
    # fy = VyC[:,:,Int(ceil(nz/2))]
    # x  = xc2[:,:,Int(ceil(nz/2))]
    # y  = yc2[:,:,Int(ceil(nz/2))]
    # display(size(x))
    # display(size(y))
    # display(size(fx))
    # display(size(fy))
    # quiver(x,y,quiver=(fx,fy))
    # u, v = rand(10),rand(10);
    # display( quiver(rand(10), gradient=(u,v)) )
    @printf("Image sliced at z index %d over nx = %d, ny = %d, nz = %d\n", Int(ceil(nz/2)), nx, ny, nz)
    # display( heatmap(transpose(T_v[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1) );
end

end

@time MainWeno()
