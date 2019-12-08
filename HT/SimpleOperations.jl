# Define working directory
if ARGS[1] == "GPU"
    println("Using GPU")
    const USE_GPU  = true      # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
else
    println("Using CPU")
    const USE_GPU  = false
end
const USE_MPI  = false
const DAT      = Float64   # Precision (Float64 or Float32)
const disksave = false     # save results to disk in binary format
include("./Macros.jl")       # Include macros - Cachemisère
using Base.Threads         # Before starting Julia do 'export JULIA_NUM_THREADS=12' (used for loops with @threads macro).
using Printf, Statistics
using LinearAlgebra
using Plots                # ATTENTION: plotting fails inside plotting library if using flag '--math-mode=fast'.
gr()

####################### Kernels Tib

@views function ComputeFlux(kx::DatArray_k,ky::DatArray_k,kz::DatArray_k, A::DatArray_k, qx::DatArray_k, qy::DatArray_k, qz::DatArray_k,
    dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)
    Dx = 1.0/dx;
    Dy = 1.0/dy;
    Dz = 1.0/dz;

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        ixi = ix+1;                                          # shifted ix for computations with inner points of an array
        iyi = iy+1;                                          # shifted iy for computations with inner points of an array
        izi = iz+1;                                          # shifted iz for computations with inner points of an array

        if (@participate_a(qx))  @all(qx) = -@all(kx)*(Dx*@d_xi(A)); end
        if (@participate_a(qy))  @all(qy) = -@all(ky)*(Dy*@d_yi(A)); end
        if (@participate_a(qz))  @all(qz) = -@all(kz)*(Dz*@d_zi(A)); end
    end
    return nothing
end

@views function UpdateT(dt::DAT, dtau::DAT, T::DatArray_k, R::DatArray_k, F::DatArray_k, qx::DatArray_k, qy::DatArray_k, qz::DatArray_k,
    dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)
    Dx = 1.0/dx;
    Dy = 1.0/dy;
    Dz = 1.0/dz;
    Dt = 1.0/dt;

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        ixi = ix+1;                                          # shifted ix for computations with inner points of an array
        iyi = iy+1;                                          # shifted iy for computations with inner points of an array
        izi = iz+1;

        if (@participate_a(F))  @all(F) = @all(R) + Dt*@inn(T) +  Dx*@d_xa(qx) + Dy*@d_ya(qy) + Dz*@d_za(qz); end
        if (@participate_i(T))  @inn(T) = @inn(T) - dtau * @all(F); end

    end
    return nothing
end

@views function UpdateP(dampx::DAT, dtau::DAT, P::DatArray_k, R::DatArray_k, F::DatArray_k, F0::DatArray_k, qx::DatArray_k, qy::DatArray_k, qz::DatArray_k,
    dx::DAT, dy::DAT, dz::DAT, nx::Integer, ny::Integer, nz::Integer)

    Dx = 1.0/dx;
    Dy = 1.0/dy;
    Dz = 1.0/dz;

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        ixi = ix+1;                                          # shifted ix for computations with inner points of an array
        iyi = iy+1;                                          # shifted iy for computations with inner points of an array
        izi = iz+1;

        if (@participate_a(F) )  @all(F ) = @all(R) +  Dx*@d_xa(qx) + Dy*@d_ya(qy) + Dz*@d_za(qz); end
        if (@participate_a(F0))  @all(F0) = @all(F) + dampx*@all(F0); end
        if (@participate_i(P ))  @inn(P ) = @inn(P) - dtau *@all(F0); end

    end
    return nothing
end

@views function SetPressureBCs(Pc_ex::DatArray_k, Pbot::DAT, Ptop::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        if (@participate_a(Pc_ex) && ix==1              ) Pc_ex[1            ,iy,iz] =          Pc_ex[2              ,iy,iz]; end
        if (@participate_a(Pc_ex) && ix==size(Pc_ex,1)  ) Pc_ex[size(Pc_ex,1),iy,iz] =          Pc_ex[size(Pc_ex,1)-1,iy,iz]; end
        if (@participate_a(Pc_ex) && iy==1              ) Pc_ex[ix            ,1,iz] = 2*Pbot - Pc_ex[ix              ,2,iz]; end
        if (@participate_a(Pc_ex) && iy==size(Pc_ex,2)  ) Pc_ex[ix,size(Pc_ex,2),iz] = 2*Ptop - Pc_ex[ix,size(Pc_ex,2)-1,iz]; end
        if (@participate_a(Pc_ex) && iz==1              ) Pc_ex[ix,iy,            1] =          Pc_ex[ix,iy,              2]; end
        if (@participate_a(Pc_ex) && iz==size(Pc_ex,3)  ) Pc_ex[ix,iy,size(Pc_ex,3)] =          Pc_ex[ix,iz,size(Pc_ex,3)-1]; end
    end
    return nothing
end


@views function SetPressureBCs_x(Pc_ex::DatArray_k, Pbot::DAT, Ptop::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        if (@participate_a(Pc_ex) && ix==1              ) Pc_ex[1            ,iy,iz] =          Pc_ex[2              ,iy,iz]; end
        if (@participate_a(Pc_ex) && ix==size(Pc_ex,1)  ) Pc_ex[size(Pc_ex,1),iy,iz] =          Pc_ex[size(Pc_ex,1)-1,iy,iz]; end
    end
    return nothing
end

@views function SetPressureBCs_y(Pc_ex::DatArray_k, Pbot::DAT, Ptop::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        if (@participate_a(Pc_ex) && iy==1              ) Pc_ex[ix            ,1,iz] = 2*Pbot - Pc_ex[ix              ,2,iz]; end
        if (@participate_a(Pc_ex) && iy==size(Pc_ex,2)  ) Pc_ex[ix,size(Pc_ex,2),iz] = 2*Ptop - Pc_ex[ix,size(Pc_ex,2)-1,iz]; end
    end
    return nothing
end

@views function SetPressureBCs_z(Pc_ex::DatArray_k, Pbot::DAT, Ptop::DAT, nx::Integer, ny::Integer, nz::Integer)

    @threadids_or_loop (nx+2,ny+2,nz+2) begin
        if (@participate_a(Pc_ex) && iz==1              ) Pc_ex[ix,iy,            1] =          Pc_ex[ix,iy,              2]; end
        if (@participate_a(Pc_ex) && iz==size(Pc_ex,3)  ) Pc_ex[ix,iy,size(Pc_ex,3)] =          Pc_ex[ix,iz,size(Pc_ex,3)-1]; end
    end
    return nothing
end


@views function AddOne(T::DatArray_k, nx::Integer, ny::Integer, nz::Integer)

    # @printf("nx_ex = %02d --- size(Tc_ex,1) = %02d\n", nx, size(T,1))

    # @threadids_or_loop (nx,ny,nz) begin
    #     ixi = ix+1;                                          # shifted ix for computations with inner points of an array
    #     iyi = iy+1;                                          # shifted iy for computations with inner points of an array
    #     izi = iz+1;
    #     if (ix<1 || ix>size(T,1) || iy<1 || iy>size(T,2) || iz<1 || iz>size(T,3)) return nothing; end
    #     # if (@participate_a(T))  @all(T) = @all(T) +2.0; end
    #     @inbounds @fastmath T[ix,iy,iz] = T[ix,iy,iz] + 12.0;
    #
    # end
    # return nothing

    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z-1) * blockDim().z + threadIdx().z
    if (ix<1 || ix>size(T,1) || iy<1 || iy>size(T,2) || iz<1 || iz>size(T,3)) return nothing; end
    @inbounds @fastmath T[ix,iy,iz] = T[ix,iy,iz] + 1.0;
    return nothing
end


#######################
## Main code
@views function HydroThermal3D()
# Visualise
Vizu = 0
# Physics
Ra          = 60;
dT          = 1.0;
Ttop        = 0.0;
Tbot        = Ttop + dT;
dP          = 5.0;
Ptop        = 0.0;
Pbot        = Ptop + dP;
T_i      = 10.0;
rho_cp   = 1.0;
lam0     = 1.0;
rad      = 1.0
xmin     = -0.0;  xmax    =      1.86038; Lx = xmax - xmin;
ymin     = -0.0;  ymax    = 1.0/4.0*xmax; Ly = ymax - ymin;
zmin     = -0.05; zmax    =         0.05; Lz = zmax - zmin;
# Numerics
nt       = 1#1000; #100; #SO: for testing; old: 10000;
nout     = 100; #1;
nx       = 2*32;
ny       = 1*32;
nz       = 3; #Int(ra*(nx-2)+2);  # Question 1: Cannot run with 1 or 3 (for making a 2D run) - 5 works                                                   #SO: conversion to Int required as ra is a float
# Preprocessing
if (USE_MPI) me, dims, nprocs, coords, comm = init_global_grid(nx, ny, nz; dimx=2, dimy=2, dimz=2);              #SO: later this can be calles as "me, = ..."
else         me, dims, nprocs, coords       = (0, [1,1,1], 1, [0,0,0]);
end
Nix      = USE_MPI ? nx_g() : nx;                                               #SO: TODO: this is obtained from the global_grid for MPI.
Niy      = USE_MPI ? ny_g() : ny;                                               #SO: this is obtained from the global_grid.
Niz      = USE_MPI ? nz_g() : nz;                                               #SO: this is obtained from the global_grid.
dx       = Lx/Nix;                                                              #SO: why not dx = Lx/(nx-1) or dx = Lx/(Nix-1) respectively
dy       = Ly/Niy;
dz       = Lz/Niz;
dt       = min(dx,dy,dz)^2/4.1;
# Initialisation
Tc_ex    = myzeros(nx+2,ny+2,nz+2);
Pc_ex    = myzeros(nx+2,ny+2,nz+2);
Ty       =  myones(nx  ,ny+1,nz  );
ktv      =  myones(nx+1,ny+1,nz+1);
kfv      =  myones(nx+1,ny+1,nz+1);
dTdy     = myzeros(nx  ,ny  ,nz  );
Rt       = myzeros(nx  ,ny  ,nz  );
Rp       = myzeros(nx  ,ny  ,nz  );
kx       = myzeros(nx+1,ny  ,nz  );
ky       = myzeros(nx  ,ny+1,nz  );
kz       = myzeros(nx  ,ny  ,nz+1);
qx       = myzeros(nx+1,ny  ,nz  )
qy       = myzeros(nx  ,ny+1,nz  )
qz       = myzeros(nx  ,ny  ,nz+1)
# lam      = myzeros(nx  ,ny  ,nz  );
# Pre-processing
if USE_MPI
    xc  = [x_g(ix,dx,Rt)+dx/2 for ix=1:nx];
    yc  = [y_g(iy,dy,Rt)+dy/2 for iy=1:ny];
    zc  = [z_g(iz,dz,Rt)+dz/2 for iz=1:nz];
    xce = [x_g(ix,dx,Rt)-dx/2 for ix=1:nx+2]; # ACHTUNG
    yce = [y_g(iy,dy,Rt)-dy/2 for iy=1:ny+2]; # ACHTUNG
    zce = [z_g(iz,dz,Rt)-dz/2 for iz=1:nz+2]; # ACHTUNG
    xv  = [x_g(ix,dx,Rt) for ix=1:nx];
    yv  = [y_g(iy,dy,Rt) for iy=1:ny];
    zv  = [z_g(iz,dz,Rt) for iz=1:nz];
    # Question 2 - how to xv
else
    xc  = LinRange(xmin+dx/2, xmax-dx/2, nx)
    yc  = LinRange(ymin+dy/2, ymax-dy/2, ny)
    zc  = LinRange(xmin+dz/2, zmax-dz/2, nz)
    xce = LinRange(xmin-dx/2, xmax+dx/2, nx+2)
    yce = LinRange(ymin-dy/2, ymax+dy/2, ny+2)
    zce = LinRange(xmin-dz/2, zmax+dz/2, nz+2)
    xv  = LinRange(xmin, xmax, nx+1)
    yv  = LinRange(ymin, ymax, ny+1)
    zv  = LinRange(xmin, zmax, nz+1)
    @printf("size xc = %d although nx = %d\n", size(xc)[1], nx)
    @printf("size xc = %d although nx = %d\n", size(xv)[1], nx+1)
    @printf("size xc = %d although nx = %d\n", size(xce)[1], nx+2)
end
# #=
(xce2,yce2,zce2) = ([x for x=xce,y=yce,z=zce], [y for x=xce,y=yce,z=zce], [z for x=xce,y=yce,z=zce]); #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
(xc2,yc2,zc2)    = ([x for x=xc,y=yc,z=zc], [y for x=xc,y=yc,z=zc], [z for x=xc,y=yc,z=zc]); #SO: Replaced [Xc,Yc,Zc] = ndgrid(xc,yc,zc); because ndgrid does not exist. Again, this can be beautifully solved with array comprehensions.
(xv2,yv2,zv2)    = ([x for x=xv,y=yv,z=zv], [y for x=xv,y=yv,z=zv], [z for x=xv,y=yv,z=zv]);
# Initial conditions: Draw Fault
kfv = Array(kfv);  # Ensure it is temporarily a CPU array
kfv[ yv2.>=2/3*Ly ] .= kfv[ yv2.>=2/3*Ly ] .+ 20;
a = -30*pi/180; b = 0.75
@. kfv[ (yv2 - xv2*a - b > 0) & (yv2 - xv2*a - (b+0.05) < 0) & (yv2>0.05)] = 20;
kfv = DatArray(kfv);
@printf("%f %f\n",maximum(kfv), minimum(kfv))
@printf("%f %f\n",maximum(yv), minimum(yv))
# Thermal field
@printf("%d %d %d\n", size(Tc_ex)[1], size(Tc_ex)[2], size(Tc_ex)[3])
@printf("%d %d %d\n", size(yc2)[1], size(yc2)[2], size(yc2)[3])
@printf("%d %d \n", Nix, nx)
# Tc_ex = Array(Tc_ex) # MAKE SURE ACTIVITY IS IN THE CPU:
# @. Tc_ex[2:end-1,2:end-1,2:end-1] = -dT/Ly * yc2 + dT
# # Tc_ex[2:end-1,2:end-1,2:end-1]    = Tc_ex[2:end-1,2:end-1,2:end-1] + rand(nx,ny,nz)/10
# @. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
# @. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:];
# @. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
# @. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2) < 0.01 ] += 0.1
# Tc_ex = DatArray(Tc_ex) # MAKE SURE ACTIVITY IS IN THE GPU
# Define kernel launch params (used only if USE_GPU set true).
# cuthreads = (32, 8, 4)
# # cublocks  = ceil.(Int, (nx+1, ny+1, nz+1)./cuthreads)
# cublocks  = ( 2, 8, 32)
# # cublocks  = ( 1, 4, 16)
cuthreads = (32, 8, 2 )
cublocks  = ( 1, 4, 16).*10

evol=[]; it1=1; time=0; warmup=3;              #SO: added warmpup; added one call to tic(); toc(); to get them compiled (to be done differently later).
## Action
# for it = it1:nt

    @printf("Thermal solver\n");
    # PT iteration parameters
    nitmax  = 1e4;
    nitout  = 1000;
    tolT    = 1e-8;
    tetT    = 0.5;
    dtauT   = tetT*min(dx,dy,dz)^2/4.1;
    Ft      = myzeros(nx  ,ny  ,nz  )
    Rt      = Array(Rt)
    ktv     = Array(ktv)
    kx      = Array(kx)
    ky      = Array(ky)
    kz      = Array(kz)
    Tc_ex   = Array(Tc_ex)
    @. Rt   = - 1.0/dt * Tc_ex[2:end-1,2:end-1,2:end-1]              # Temperature RHS
    @. kx   = 0.25*(ktv[:,1:end-1,1:end-1] + ktv[:,1:end-1,2:end-0] + ktv[:,2:end-0,1:end-1] + ktv[:,2:end-0,2:end-0])
    @. ky   = 0.25*(ktv[1:end-1,:,1:end-1] + ktv[1:end-1,:,2:end-0] + ktv[2:end-0,:,1:end-1] + ktv[2:end-0,:,2:end-0])
    @. kz   = 0.25*(ktv[1:end-1,1:end-1,:] + ktv[1:end-1,2:end-0,:] + ktv[2:end-0,1:end-1,:] + ktv[2:end-0,2:end-0,:])
    Rt      = DatArray(Rt)
    ktv     = DatArray(ktv)
    kx      = DatArray(kx)
    ky      = DatArray(ky)
    kz      = DatArray(kz)
    Tc_ex   = DatArray(Tc_ex)

    # CPU
    if ARGS[1] == "CPU"
        Tc_ex = Array(Tc_ex)
        @. Tc_ex += 1.0
        Tc_ex = DatArray(Tc_ex)
    end

    if ARGS[1] == "GPU"

        @printf("nx_ex = %02d --- size(Tc_ex,1) = %02d\n", nx+2, size(Tc_ex,1))
        @kernel cublocks cuthreads AddOne(Tc_ex, nx+2, ny+2, nz+2);                       @devicesync();
    end

    # for iter = 1:nitmax
    #     @kernel cublocks cuthreads SetPressureBCs_x(Tc_ex, Tbot, Ttop, nx, ny, nz);                       @devicesync();
    #     @kernel cublocks cuthreads SetPressureBCs_y(Tc_ex, Tbot, Ttop, nx, ny, nz);                       @devicesync();
    #     @kernel cublocks cuthreads SetPressureBCs_z(Tc_ex, Tbot, Ttop, nx, ny, nz);                       @devicesync();
    #     @kernel cublocks cuthreads ComputeFlux(kx, ky, kz, Tc_ex, qx, qy, qz, dx, dy, dz, nx, ny, nz);    @devicesync();
    #     @kernel cublocks cuthreads UpdateT(dt, dtauT, Tc_ex, Rt, Ft, qx, qy, qz, dx, dy, dz, nx, ny, nz); @devicesync();
    #     if (USE_MPI) update_halo!(Tc_ex); end
    #     if mod(iter,nitout) == 1
    #         # nFt = norm(Ft)/sqrt(nx*ny*nz) # Question 3 - how to norm with GPU and MPI
    #         nFt = mean_g(abs.(Ft[:]))/sqrt(nx*ny*nz);
    #         if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
    #         if nFt<tolT
    #             break
    #         end
    #     end
    # end

    Rt      = Array(Rt)
    Tc_ex   = Array(Tc_ex)
    @printf("min(Rt)    = %02.4e - max(Rt)    = %02.4e\n", minimum(Rt),    maximum(Rt)    )
    @printf("min(Tc_ex) = %02.4e - max(Tc_ex) = %02.4e\n", minimum(Tc_ex), maximum(Tc_ex) )

    show(stdout, "text/plain", Tc_ex[:,:,2])
    Rt      = DatArray(Rt)
    Tc_ex   = DatArray(Tc_ex)

if (USE_MPI) finalize_global_grid(); end

# =#

end # Diffusion3D()

@time HydroThermal3D()
