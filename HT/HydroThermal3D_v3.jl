# Define working directory
cd("/Users/tduretz/Dropbox/Julia/Thib/DiffusionLudoTest")
const USE_GPU  = false     # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const USE_MPI  = false
const DAT      = Float64   # Precision (Float64 or Float32)
const disksave = false     # save results to disk in binary format
include("Macros.jl")       # Include macros - Cachemisère
using Base.Threads         # Before starting Julia do 'export JULIA_NUM_THREADS=12' (used for loops with @threads macro).
using Printf, Statistics
using LinearAlgebra
using Plots                # ATTENTION: plotting fails inside plotting library if using flag '--math-mode=fast'.
pyplot()

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
    return nothing # Question 4, is that necessary?
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
        if (ix==1   ) Pc_ex[ix,iy,iz] =          Pc_ex[ix+1,iy,iz]; end
        if (ix==nx+2) Pc_ex[ix,iy,iz] =          Pc_ex[ix-1,iy,iz]; end
        if (iy==1   ) Pc_ex[ix,iy,iz] = 2*Pbot - Pc_ex[ix,iy+1,iz]; end
        if (iy==ny+2) Pc_ex[ix,iy,iz] = 2*Ptop - Pc_ex[ix,iy-1,iz]; end
        if (iz==1   ) Pc_ex[ix,iy,iz] =          Pc_ex[ix,iy,iz+1]; end
        if (iz==nz+2) Pc_ex[ix,iy,iz] =          Pc_ex[ix,iy,iz-1]; end
    end
    return nothing
end

#######################
## Main code
@views function HydroThermal3D()
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
ny       = 2*32;
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

@. Tc_ex[2:end-1,2:end-1,2:end-1] = -dT/Ly * yc2 + dT
Tc_ex[2:end-1,2:end-1,2:end-1]    = Tc_ex[2:end-1,2:end-1,2:end-1] + rand(nx,ny,nz)/10
@. Tc_ex[1,:,:] =          Tc_ex[2,:,:]; @. Tc_ex[end,:,:] =          Tc_ex[end-1,:,:];
@. Tc_ex[:,1,:] = 2*Tbot - Tc_ex[:,2,:]; @. Tc_ex[:,end,:] = 2*Ttop - Tc_ex[:,end-1,:];
@. Tc_ex[:,:,1] =          Tc_ex[:,:,1]; @. Tc_ex[:,:,end] =          Tc_ex[:,:,end-1];
@. Tc_ex[ ((xce2-xmax/2)^2 + (yce2-ymax/2)^2) < 0.01 ] += 0.1

# Define kernel launch params (used only if USE_GPU set true).
cuthreads = (32, 8, 4)
# cublocks  = ceil.(Int, (nx+1, ny+1, nz+1)./cuthreads)
cublocks  = ( 2, 8, 32)
# cublocks  = ( 1, 4, 16)

# # Preparation of visualisation
# # ENV["GKSwstype"]="nul"
# # anim    = Animation();
# loadpath = "./Figures/"
# anim = Animation(loadpath,String[]);
# if me == 0
#     println("Animation directory: $(anim.dir)");                # LR: print path to temp dir for animation
#     # LR: to do: remove the temp folder after animation and erase it or move it to same folder as gif
# end
# isave   = 0;
# red     = 1;                                                # Factor by wich size is reduced for visualization
# nx_v    = (nx-2)÷red*dims[1];
# ny_v    = (ny-2)÷red*dims[2];
# nz_v    = (nz-2)÷red*dims[3];
#
# T_v   = zeros(nx_v, ny_v, nz_v);
# T_inn = zeros(nx-2, ny-2, nz-2);

evol=[]; it1=1; time=0; warmup=3;  #tic(); toc();            #SO: added warmpup; added one call to tic(); toc(); to get them compiled (to be done differently later).
## Action
for it = it1:nt

    @printf("Thermal solver\n")
    # PT iteration parameters
    nitmax   = 1e4;
    nitout   = 1000;
    tolT     = 1e-8;
    tetT    = 0.5;
    dtauT   = tetT*min(dx,dy,dz)^2/4.1;
    Ft      = myzeros(nx  ,ny  ,nz  )
    @. Rt   = - 1.0/dt * Tc_ex[2:end-1,2:end-1,2:end-1]              # Temperature RHS
    @. kx   = 0.25*(ktv[:,1:end-1,1:end-1] + ktv[:,1:end-1,2:end-0] + ktv[:,2:end-0,1:end-1] + ktv[:,2:end-0,2:end-0])
    @. ky   = 0.25*(ktv[1:end-1,:,1:end-1] + ktv[1:end-1,:,2:end-0] + ktv[2:end-0,:,1:end-1] + ktv[2:end-0,:,2:end-0])
    @. kz   = 0.25*(ktv[1:end-1,1:end-1,:] + ktv[1:end-1,2:end-0,:] + ktv[2:end-0,1:end-1,:] + ktv[2:end-0,2:end-0,:])
    for iter = 1:nitmax
        @kernel cublocks cuthreads SetPressureBCs(Tc_ex, Tbot, Ttop, nx, ny, nz);                         @devicesync();
        @kernel cublocks cuthreads ComputeFlux(kx, ky, kz, Tc_ex, qx, qy, qz, dx, dy, dz, nx, ny, nz);    @devicesync();
        @kernel cublocks cuthreads UpdateT(dt, dtauT, Tc_ex, Rt, Ft, qx, qy, qz, dx, dy, dz, nx, ny, nz); @devicesync();
        if (USE_MPI) update_halo!(Tc_ex); end
        if mod(iter,nitout) == 1
            # nFt = norm(Ft)/sqrt(nx*ny*nz) # Question 3 - how to norm with GPU and MPI
            nFt = mean_g(abs.(Ft[:]))/sqrt(nx*ny*nz);
            if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
            if nFt<tolT
                break
            end
        end
    end

    @printf("Darcy solver\n")
    # PT iteration parameters
    nitmax   = 1e4;
    nitout   = 1000;
    tolP     = 1e-10;
    tetP     = 1/10;
    dtauP    = tetP/4.1*min(dx,dy,dz)^2;
    Pdamp    = 1.25;
    dampx    = 1*(1-Pdamp/nx);
    Ft       = myzeros(nx  ,ny  ,nz  )
    Ft0      = myzeros(nx  ,ny  ,nz  )
    ExCentroid2VyOnCPU!( Ty, Tc_ex )      # Interpolate T on Vy points
    @. Rp   = Ra/dy * (0.5*(kfv[1:end-1,2:end,1:end-1] + kfv[2:end,2:end,2:end])*Ty[:,2:end,:] - 0.5*(kfv[1:end-1,1:end-1,1:end-1] + kfv[2:end,1:end-1,2:end])*Ty[:,1:end-1,:])
    @. kx   = 0.25*(kfv[:,1:end-1,1:end-1] + kfv[:,1:end-1,2:end-0] + kfv[:,2:end-0,1:end-1] + kfv[:,2:end-0,2:end-0])
    @. ky   = 0.25*(kfv[1:end-1,:,1:end-1] + kfv[1:end-1,:,2:end-0] + kfv[2:end-0,:,1:end-1] + kfv[2:end-0,:,2:end-0])
    @. kz   = 0.25*(ktv[1:end-1,1:end-1,:] + kfv[1:end-1,2:end-0,:] + kfv[2:end-0,1:end-1,:] + kfv[2:end-0,2:end-0,:])
    for iter = 1:nitmax
        @kernel cublocks cuthreads SetPressureBCs(Pc_ex, Pbot, Ptop, nx, ny, nz);                                 @devicesync();
        @kernel cublocks cuthreads ComputeFlux(kx, ky, kz, Pc_ex, qx, qy, qz, dx, dy, dz, nx, ny, nz);            @devicesync();
        @kernel cublocks cuthreads UpdateP(dampx, dtauP, Pc_ex, Rp, Ft, Ft0, qx, qy, qz, dx, dy, dz, nx, ny, nz); @devicesync();
        if (USE_MPI) update_halo!(Pc_ex); end
        if mod(iter,nitout) == 1
            nFt = mean_g(abs.(Ft[:]))/sqrt(nx*ny*nz);
            if (me==0) @printf("PT iter. #%05d - || Ft || = %2.2e\n", iter, nFt) end
            if nFt<tolP
                break
            end
        end
    end
    time  = time + dt;
    @printf("\n-> it=%d, time=%.1e, dt=%.1e, \n", it, time, dt);

    X = Pc_ex[2:end-1,2:end-1,2:end-1]
    display( heatmap(xc, yc, transpose(X[:,:,Int(ceil(nz/2))]),c=:viridis,aspect_ratio=1) );
    # display(  contourf(xc,yc,transpose(Ty[:,:,Int(ceil(nz/2))])) ) # accede au sublot 111
    #quiver(x,y,(f,f))
    @printf("Imaged sliced at z index %d over nx = %d, ny = %d, nz = %d\n", Int(ceil(nz/2)), nx, ny, nz)
    @printf("dx = %f, dy = %f, dz=%f\n", dx, dy, dz) 
    # display( heatmap(transpose(T_v[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1) );



#     # Postprocessing
#     if mod(it,nout)==0
#         T_max = maximum_g(abs.(T[:] ));
#         if me == 0
#             @printf("\n-> it=%d, time=%.1e, dt=%.1e, max(T)=%.3e \n", it, time, dt, T_max);
#         end
#         push!(evol, [it, maximum_g(abs.(T[:]))]); #SO: Replaced evol=[evol; ...] as it cannot append concatenate matrices/vectors if they do not have the same dimensions (evol is empty at the beginning...); ...
#         # Visualization
#         if USE_MPI
#             T_inn .= inn(T);    # Take inner points and reduce size by factor red in each dimension if desired (maybe red should be just 1 or 2 always; if red == 2, then we reduce the total size already by a factor 8! :) ).
#             gather!(T_inn, T_v); # Gather (reduced sized) data
#             if me == 0
#                 heatmap(transpose(T_v[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1); frame(anim);
#                 if disksave
#                     SaveArray("T",T_v,isave);
#                 end
#             end
#         else
#             # display(heatmap( transpose(Array(phi)[:,16,:]),c=:viridis,aspect_ratio=1))
#             heatmap(transpose(Array(T)[:,Int(ceil(ny_v/2)),:]),c=:viridis,aspect_ratio=1); frame(anim);
#             if disksave
#                 SaveArray("T",T,isave);
#             end
#         end
#         isave = isave + 1;
#     end#nout
end#it
#
# if me == 0
#     # println("time_s=$time_s T_eff=$T_eff niter=$niter iterMin=$iterMin iterMax=$iterMax");
#     # println("nprocs $nprocs dims $(dims[1]) $(dims[2]) $(dims[3]) fdims 1 1 1 nxyz $nx $ny $nz nt $(iterMin-warmup) nb_it 1 PRECIS $(sizeof(DAT)) time_s $time_s block $(cuthreads[1]) $(cuthreads[2]) $(cuthreads[3]) grid $(cublocks[1]) $(cublocks[2]) $(cublocks[3])\n");
#     gif(anim, "Diffusion_fps15_1.gif", fps=15);
# end
if (USE_MPI) finalize_global_grid(); end

# =#

end # Diffusion3D()

HydroThermal3D()
