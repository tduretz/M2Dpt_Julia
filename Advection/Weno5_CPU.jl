const vectorized = 0
const Upwind     = 0
const USE_GPU    = false
const USE_MPI    = false
const DAT        = Float64   # Precision (Float64 or Float32)
include("../Macros.jl")
# include("./AdvectionSchemes.jl")
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
    end
    return nothing
end

@views function dFdx_Weno5(dFdxi::DatArray_k, V1::DatArray_k, V2::DatArray_k, V3::DatArray_k, V4::DatArray_k, V5::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+0,ny+0,nz+0) begin
        v1 = V1[ix,iy,iz]
        v2 = V2[ix,iy,iz]
        v3 = V3[ix,iy,iz]
        v4 = V4[ix,iy,iz]
        v5 = V5[ix,iy,iz]
        p1   = v1/3.0 - 7.0/6.0*v2 + 11.0/6.0*v3
        p2   =-v2/6.0 + 5.0/6.0*v3 + v4/3.0
        p3   = v3/3.0 + 5.0/6.0*v4 - v5/6.0
        maxV = max(max(max(max(v1^2,v2^2), v3^2), v4^2), v5^2)
        e    = 10^(-99) + 1e-6*maxV
        w1    = 13.0/12.0*(v1-2.0*v2+v3)^2.0 + 1.0/4.0*(v1-4.0*v2+3.0*v3)^2.0;
        w2    = 13.0/12.0*(v2-2.0*v3+v4)^2.0 + 1.0/4.0*(v2-v4)^2.0;
        w3    = 13.0/12.0*(v3-2.0*v4+v5)^2.0 + 1.0/4.0*(3.0*v3-4.0*v4+v5)^2.0;
        w1    = 0.1/(w1+e)^2.0;
        w2    = 0.6/(w2+e)^2.0;
        w3    = 0.3/(w3+e)^2.0;
        w     = (w1+w2+w3)
        w1    = w1/w;
        w2    = w2/w;
        w3    = w3/w;
        # dFdxi[ix,iy,iz] = w1*p1 + w2*p2 + w3*p3
        if @participate_a(dFdxi) @all(dFdxi)= w1*p1 + w2*p2 + w3*p3; end
    end
return nothing
end

@views function Gradients_minus_x_Weno5(v1::DatArray_k, v2::DatArray_k, v3::DatArray_k, v4::DatArray_k, v5::DatArray_k, Fc_exxx::DatArray_k, dx::Float64, dy::Float64, dz::Float64, nx::Integer, ny::Integer, nz::Integer)

    if vectorized == 1
        # Vectorized style - slow but faster than @threadids_or_loop
        @. v1    = 1.0/dx*(Fc_exxx[2:end-5,4:end-3,4:end-3]-Fc_exxx[1:end-6,4:end-3,4:end-3]);
        @. v2    = 1.0/dx*(Fc_exxx[3:end-4,4:end-3,4:end-3]-Fc_exxx[2:end-5,4:end-3,4:end-3]);
        @. v3    = 1.0/dx*(Fc_exxx[4:end-3,4:end-3,4:end-3]-Fc_exxx[3:end-4,4:end-3,4:end-3]);
        @. v4    = 1.0/dx*(Fc_exxx[5:end-2,4:end-3,4:end-3]-Fc_exxx[4:end-3,4:end-3,4:end-3]);
        @. v5    = 1.0/dx*(Fc_exxx[6:end-1,4:end-3,4:end-3]-Fc_exxx[5:end-2,4:end-3,4:end-3]);
    else
        @threadids_or_loop (nx+6,ny+6,nz+6) begin
            ixiii = ix + 3
            iyiii = iy + 3
            iziii = iz + 3
            if @participate_a(v1) @all(v1) = 1.0/dx*( @in_xxx_xm2(Fc_exxx) -  @in_xxx_xm3(Fc_exxx) ); end
            if @participate_a(v2) @all(v2) = 1.0/dx*( @in_xxx_xm1(Fc_exxx) -  @in_xxx_xm2(Fc_exxx) ); end
            if @participate_a(v3) @all(v3) = 1.0/dx*(     @in_xxx(Fc_exxx) -  @in_xxx_xm1(Fc_exxx) ); end
            if @participate_a(v4) @all(v4) = 1.0/dx*( @in_xxx_xp1(Fc_exxx) -      @in_xxx(Fc_exxx) ); end
            if @participate_a(v5) @all(v5) = 1.0/dx*( @in_xxx_xp2(Fc_exxx) -  @in_xxx_xp1(Fc_exxx) ); end
        end
    end
return nothing
end

############################################################

@views function MainWeno()
# General
Vizu   = 1
order  = 2;
nit    = 50;
# Domain
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
# Grid
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
# Initial conditions
Vx       =   myones(nx+1,ny+0,nz+0);
Vy       =  myzeros(nx+0,ny+1,nz+0);
Vz       =  myzeros(nx+0,ny+0,nz+1);
Tc       =  myzeros(nx+0,ny+0,nz+0);
xC       = 0.1
yC       = 0.5*(ymin+ymax)
zC       = 0.5*(zmin+zmax)
@. Tc    = exp(-(xc2-xC)^2/ 0.001 - (yc2-yC)^2/ 0.001 - (zc2-zC)^2/ 0.001)
# Compute Courant criteria
dt = 0.25*min(dx,dy,dz) / max( maximum_g(Vx), maximum_g(Vy), maximum_g(Vz))
# Upwind velocities
Vxm     =  myzeros(nx+0,ny+0,nz+0);
Vxp     =  myzeros(nx+0,ny+0,nz+0);
VxPlusMinus(Vxm, Vxp, Vx, nx, ny, nz)
Vym     =  myzeros(nx+0,ny+0,nz+0);
Vyp     =  myzeros(nx+0,ny+0,nz+0);
VyPlusMinus(Vym, Vyp, Vy, nx, ny, nz)
Vzm     =  myzeros(nx+0,ny+0,nz+0);
Vzp     =  myzeros(nx+0,ny+0,nz+0);
VzPlusMinus(Vzm, Vzp, Vz, nx, ny, nz)
# Pre-processing
Tc_exxx =  myzeros(nx+6,ny+6,nz+6)
Tc_ex   =  myzeros(nx+2,ny+2,nz+2)
v1      =  myzeros(nx+0,ny+0,nz+0);
v2      =  myzeros(nx+0,ny+0,nz+0);
v3      =  myzeros(nx+0,ny+0,nz+0);
v4      =  myzeros(nx+0,ny+0,nz+0);
v5      =  myzeros(nx+0,ny+0,nz+0);
dTdxp   =  myzeros(nx+0,ny+0,nz+0);
dTdxm   =  myzeros(nx+0,ny+0,nz+0);
Told    =  myzeros(nx+0,ny+0,nz+0);
time   = 0
# Time loop
for it=1:nit

    time += dt

    if Upwind == 0 # --- WENO-5
        @. Told = Tc;
        for io=1:order
            # Weno 5
            @. Tc_exxx[4:end-3,4:end-3,4:end-3] = Tc;
            @. Tc_exxx[3      ,4:end-3,4:end-3] = Tc[1  ,:,:]; @. Tc_exxx[2    ,4:end-3,4:end-3] =  Tc[1  ,:,:];; @. Tc_exxx[1    ,4:end-3,4:end-3] =  Tc[  1,:,:];
            @. Tc_exxx[end    ,4:end-3,4:end-3] = Tc[end,:,:]; @. Tc_exxx[end-1,4:end-3,4:end-3] =  Tc[end,:,:];; @. Tc_exxx[end-2,4:end-3,4:end-3] =  Tc[end,:,:];
            @kernel cublocks cuthreads Gradients_minus_x_Weno5(v1, v2, v3, v4, v5, Tc_exxx, dx, dy, dz, nx, ny, nz); @devicesync();
            @kernel cublocks cuthreads dFdx_Weno5(dTdxm, v1, v2, v3, v4, v5, nx, ny, nz); @devicesync();
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
            @. Tc_ex[2:end-1,2:end-1,2:end-1] = Tc
            @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
            @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
            @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
            @. dTdxp = 1.0/dx*(Tc_ex[3:end-0,2:end-1,2:end-1] - Tc_ex[2:end-1,2:end-1,2:end-1])
            @. dTdxm = 1.0/dx*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[1:end-2,2:end-1,2:end-1])
            @. Tc    = Tc - dt*(Vxp*dTdxm + Vxm*dTdxp);
        end
        @. Tc = (1.0/order)*Tc + (1.0-(1.0/order))*Told;

        # @. Told = Tc
        # for io=1:order
        #     @. Tc_ex[2:end-1,2:end-1,2:end-1] = Tc
        #     @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
        #     @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
        #     @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
        #     @. dTdxp = 1.0/dy*(Tc_ex[2:end-1,3:end-0,2:end-1] - Tc_ex[2:end-1,2:end-1,2:end-1])
        #     @. dTdxm = 1.0/dy*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[2:end-1,1:end-2,2:end-1])
        #     @. Tc    = Tc - dt*(Vyp*dTdxm + Vym*dTdxp);
        # end
        # @. Tc = (1.0/order)*Tc + (1.0-(1.0/order))*Told;
        #
        # @. Told = Tc;
        # for io=1:order
        #     @. Tc_ex[2:end-1,2:end-1,2:end-1] = Tc
        #     @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
        #     @. Tc_ex[:,1,:] =          Tc_ex[:,2,:]; @. Tc_ex[:,end,:] =          Tc_ex[:,end-1,:];
        #     @. Tc_ex[:,:,1] =          Tc_ex[:,:,2]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
        #     @. dTdxp = 1.0/dz*(Tc_ex[2:end-1,2:end-1,3:end-0] - Tc_ex[2:end-1,2:end-1,2:end-1])
        #     @. dTdxm = 1.0/dz*(Tc_ex[2:end-1,2:end-1,2:end-1] - Tc_ex[2:end-1,2:end-1,1:end-2])
        #     @. Tc    = Tc - dt*(Vzp*dTdxm + Vzm*dTdxp);
        # end
        # @. Tc = (1.0/order)*Tc + (1.0-1.0/order)*Told;
    end
end

display(xC + time*1.0)
display(yC + time*1.0)

if (Vizu == 1)
    X = Tc
    display( heatmap(xc, yc, transpose(X[:,:,Int(ceil(nz/2))]),c=:viridis,aspect_ratio=1) );
    @printf("Image sliced at z index %d over nx = %d, ny = %d, nz = %d\n", Int(ceil(nz/2)), nx, ny, nz)
end

end

@time MainWeno()
