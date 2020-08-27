# Visco-elastic compressible formulation
const USE_GPU  = true      # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID   = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra  # ATTENTION: plotting fails inside plotting library if using flag '--math-mode=fast'.
gr()

################################ Macros from cuda_scientific
import ParallelStencil: INDICES
ix,  iy   = INDICES[1], INDICES[2]
ixi, iyi  = :($ix+1), :($iy+1)
macro av_xya(A) esc(:( ($A[$ix, $iy] + $A[($ix+1), $iy   ] + $A[$ix,($iy+1)] + $A[($ix+1),($iy+1)])*0.25 )) end
macro  d_yxi(A) esc(:( $A[ $ixi  ,($iy+1)] - $A[ $ixi  ,$iy  ] )) end
macro  d_xyi(A) esc(:( $A[($ix+1), $iyi  ] - $A[ $ix   ,$iyi ] )) end

function save_array(A, A_name, isave)
	fid=open("$(isave)_$(A_name).res", "w"); write(fid, Array(A)); close(fid);
end
@static if USE_GPU
	macro sqrt(args...) esc(:(CUDAnative.sqrt($(args...)))) end
else
	macro sqrt(args...) esc(:(Base.sqrt($(args...)))) end
end
############################################################################################################

@parallel_indices (ix,iy) function weights!(wSW::Data.Array, wSE::Data.Array, wNW::Data.Array, wNE::Data.Array)

	nxA = size(wSW,1)
	nyA = size(wSW,2)

	if (ix==nxA && iy<=nyA)  wSW[nxA, iy ] = 2; end # wSW(end,:) = 2;
	if (ix<=nxA && iy==nyA)  wSW[ix , nyA] = 2; end # wSW(:,end) = 2;
	if (ix==nxA && iy==nyA)  wSW[nxA, nyA] = 4; end # wSW(end,end) = 4;
	if (ix==1   && iy<=nyA)  wSW[1  , iy ] = 0; end # wSW(1,:) = 0;
	if (ix<=nxA && iy==1  )  wSW[ix , 1  ] = 0; end # wSW(:,1) = 0;

	if (ix==1   && iy<=nyA)  wSE[1  , iy ] = 2; end # wSE(1,:) = 2;
	if (ix<=nxA && iy==nyA)  wSE[ix , nyA] = 2; end # wSE(:,end) = 2;
	if (ix==1   && iy==nyA)  wSE[1  , nyA] = 4; end # wSE(1,end) = 4;
	if (ix==nxA && iy<=nyA)  wSE[nxA, iy ] = 0; end # wSE(end,:) = 0;
	if (ix<=nxA && iy==1  )  wSE[ix , 1  ] = 0; end # wSE(:,1) = 0;

	if (ix==nxA && iy<=nyA)  wNW[nxA, iy ] = 2; end # wNW(end,:) = 2;
	if (ix<=nxA && iy==1  )  wNW[ix , 1  ] = 2; end # wNW(:,1) = 2;
	if (ix==nxA && iy==1  )  wNW[nxA, 1  ] = 4; end # wNW(end,1) = 4;
	if (ix==1   && iy<=nyA)  wNW[1  , iy ] = 0; end # wNW(1,:) = 0;
	if (ix<=nxA && iy==nyA)  wNW[ix , nyA] = 0; end # wNW(:,end) = 0;

    if (ix==1   && iy<=nyA)  wNE[1  , iy ] = 2; end # wNE(1,:) = 2;
    if (ix<=nxA && iy==1  )  wNE[ix , 1  ] = 2; end # wNE(:,1) = 2;
    if (ix==1   && iy==1  )  wNE[1  , 1  ] = 4; end # wNE(1,1) = 4;
    if (ix==nxA && iy<=nyA)  wNE[nxA, iy ] = 0; end # wNE(end,:) = 0;
    if (ix<=nxA && iy==nyA)  wNE[ix , nyA] = 0; end # wNE(:,end) = 0;

    return
end

@parallel_indices (ix,iy) function c2v!(Av::Data.Array, Ac::Data.Array, AvSW::Data.Array, AvSE::Data.Array, AvNW::Data.Array, AvNE::Data.Array, wSW::Data.Array, wSE::Data.Array, wNW::Data.Array, wNE::Data.Array)

	if (ix>=2 && ix<=size(AvSW,1)   && iy>=2 && iy<=size(AvSW,2)  )  AvSW[ix, iy] = Ac[ix-1,iy-1]; end
	if (ix>=1 && ix<=size(AvSE,1)-1 && iy>=2 && iy<=size(AvSE,2)  )  AvSE[ix, iy] = Ac[ix  ,iy-1]; end
	if (ix>=2 && ix<=size(AvNW,1)   && iy>=1 && iy<=size(AvNW,2)-1)  AvNW[ix, iy] = Ac[ix-1,iy  ]; end
	if (ix>=1 && ix<=size(AvNE,1)-1 && iy>=1 && iy<=size(AvNE,2)-1)  AvNE[ix, iy] = Ac[ix  ,iy  ]; end

	if (ix<=size(Av,1) && iy<=size(Av,2))  Av[ix, iy] = 0.25*(wSW[ix, iy]*AvSW[ix, iy] + wSE[ix, iy]*AvSE[ix, iy] + wNW[ix, iy]*AvNW[ix, iy] + wNE[ix, iy]*AvNE[ix, iy] ); end

	return
end

@parallel function v2c!(Ac::Data.Array, Av::Data.Array)

	@all(Ac) = @av_xya(Av)

	return
end

@parallel function reset!(λc::Data.Array, λv::Data.Array)

	@all(λc) = 0.0
	@all(λv) = 0.0

	return
end

@parallel function swap0ld!(τxxc::Data.Array, τxxc0::Data.Array, τyyc::Data.Array, τyyc0::Data.Array, τzzc::Data.Array, τzzc0::Data.Array, τxyv::Data.Array, τxyv0::Data.Array, Ptc::Data.Array, Ptc0::Data.Array, τxxv::Data.Array, τxxv0::Data.Array, τyyv::Data.Array, τyyv0::Data.Array, τzzv::Data.Array, τzzv0::Data.Array, τxyc::Data.Array, τxyc0::Data.Array, Ptv::Data.Array, Ptv0::Data.Array)

    @all(τxxc0) = @all(τxxc)
    @all(τyyc0) = @all(τyyc)
	@all(τzzc0) = @all(τzzc)
    @all(τxyv0) = @all(τxyv)
    @all(Ptc0)  = @all(Ptc)
    @all(τxxv0) = @all(τxxv)
    @all(τyyv0) = @all(τyyv)
	@all(τzzv0) = @all(τzzv)
    @all(τxyc0) = @all(τxyc)
    @all(Ptv0)  = @all(Ptv)

    return
end

@parallel function timesteps!(scPt::Data.Number, scV::Data.Number, min_dxy2::Data.Number, max_nxy::Int, dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, η_vec::Data.Array)

    @all(dτVx) = min_dxy2/(@av_xa(η_vec))/4.1/scV
    @all(dτVy) = min_dxy2/(@av_ya(η_vec))/4.1/scV
    @all(dτPt) = 4.1*@all(η_vec)/max_nxy/2.1/scPt

    return
end

@parallel_indices (ix,iy) function setBCvel!(Vxe::Data.Array, Vye::Data.Array)
	# cp macro sets free-slip boundary conditions for extended velocity array.
    # if ( ix<=size(Vxe,1)    && iy<=(size(Vxe,2)-2)) Vxe[ix  ,iy+1] = Vx[ix,iy]; end
    # if ((ix<=size(Vye,1)-2) && iy<= size(Vye,2)  )  Vye[ix+1,iy  ] = Vy[ix,iy]; end

    if (ix<=size(Vxe,1) && iy<=size(Vxe,2)) Vxe[ix         , 1          ] = Vxe[ix           , 2            ]; end
    if (ix<=size(Vxe,1) && iy<=size(Vxe,2)) Vxe[ix         , size(Vxe,2)] = Vxe[ix           , size(Vxe,2)-1]; end
    if (ix<=size(Vye,1) && iy<=size(Vye,2)) Vye[1          , iy         ] = Vye[2            , iy           ]; end
    if (ix<=size(Vye,1) && iy<=size(Vye,2)) Vye[size(Vye,1), iy         ] = Vye[size(Vye,1)-1, iy           ]; end

    return
end

@parallel function compute_ε!(Vxe::Data.Array, Vye::Data.Array, εxxc::Data.Array, εyyc::Data.Array, εxyv::Data.Array, εzzc::Data.Array, divVc::Data.Array,
	                          _dx::Data.Number, _dy::Data.Number )
	# Deviatoric strain rate tensor
	@all(divVc) = _dx*@d_xyi(Vxe) + _dy*@d_yxi(Vye)
	@all(εxxc)  = _dx*@d_xyi(Vxe) - 1.0/3.0*@all(divVc)
	@all(εyyc)  = _dy*@d_yxi(Vye) - 1.0/3.0*@all(divVc)
	@all(εzzc)  =                 - 1.0/3.0*@all(divVc)
	@all(εxyv)  = 0.5*(_dy*@d_ya(Vxe) + _dx*@d_xa(Vye))

	return
end

@parallel function compute_τ!(εxxc::Data.Array, εyyc::Data.Array, εxyv::Data.Array, εzzc::Data.Array, εxxv::Data.Array, εyyv::Data.Array, εxyc::Data.Array, εzzv::Data.Array, τxxc::Data.Array, τyyc::Data.Array, τxyv::Data.Array, τzzc::Data.Array, τxxv::Data.Array, τyyv::Data.Array, τxyc::Data.Array, τzzv::Data.Array, τxxc0::Data.Array, τyyc0::Data.Array, τzzc0::Data.Array, τxyv0::Data.Array, τxxv0::Data.Array, τyyv0::Data.Array, τzzv0::Data.Array, τxyc0::Data.Array, η_vec::Data.Array, η_vev::Data.Array, η_ec::Data.Array, η_ev::Data.Array)
	# Deviatoric stress tensor
	@all(τxxc)  = 2.0*@all(η_vec)*(@all(εxxc) + @all(τxxc0)/(2.0*@all(η_ec)))
	@all(τyyc)  = 2.0*@all(η_vec)*(@all(εyyc) + @all(τyyc0)/(2.0*@all(η_ec)))
	@all(τzzc)  = 2.0*@all(η_vec)*(@all(εzzc) + @all(τzzc0)/(2.0*@all(η_ec)))
	@all(τxyv)  = 2.0*@all(η_vev)*(@all(εxyv) + @all(τxyv0)/(2.0*@all(η_ev)))

	@all(τxxv)  = 2.0*@all(η_vev)*(@all(εxxv) + @all(τxxv0)/(2.0*@all(η_ev)))
	@all(τyyv)  = 2.0*@all(η_vev)*(@all(εyyv) + @all(τyyv0)/(2.0*@all(η_ev)))
	@all(τzzv)  = 2.0*@all(η_vev)*(@all(εzzv) + @all(τzzv0)/(2.0*@all(η_ev)))
	@all(τxyc)  = 2.0*@all(η_vec)*(@all(εxyc) + @all(τxyc0)/(2.0*@all(η_ec)))

	return
end

@parallel function check_yield!(K::Data.Number, dt::Data.Number, sin_ψ::Data.Number, sin_ϕ::Data.Number, cos_ϕ::Data.Number, C::Data.Number, η_vp::Data.Number, Ptc::Data.Array, Ptv::Data.Array, λc::Data.Array, λv::Data.Array, Fc::Data.Array, Fv::Data.Array, τiic::Data.Array, τiiv::Data.Array, τxxc::Data.Array, τyyc::Data.Array, τzzc::Data.Array, τxyc::Data.Array, τxxv::Data.Array, τyyv::Data.Array, τzzv::Data.Array, τxyv::Data.Array)
	# Rheology
	@all(τiic) = sqrt( 0.5*(@all(τxxc)*@all(τxxc) + @all(τyyc)*@all(τyyc) + @all(τzzc)*@all(τzzc)) + @all(τxyc)*@all(τxyc) )
	@all(τiiv) = sqrt( 0.5*(@all(τxxv)*@all(τxxv) + @all(τyyv)*@all(τyyv) + @all(τzzv)*@all(τzzv)) + @all(τxyv)*@all(τxyv) )

	# Check yield
	@all(Fc) = @all(τiic) - (@all(Ptc) + K*dt*sin_ψ*@all(λc))*sin_ϕ - C*cos_ϕ - η_vp*@all(λc)
	@all(Fv) = @all(τiiv) - (@all(Ptv) + K*dt*sin_ψ*@all(λv))*sin_ϕ - C*cos_ϕ - η_vp*@all(λv)

	return
end

@parallel_indices (ix,iy) function plastic_nodes!(Fc::Data.Array, Fv::Data.Array, Plc::Data.Array, Plv::Data.Array)
	# reset plastic node flags
	if (ix<=size(Plc,1) && iy<=size(Plc,2)) Plc[ix,iy] = 0.0; end
	if (ix<=size(Plv,1) && iy<=size(Plv,2)) Plv[ix,iy] = 0.0; end

	if (ix<=size(Fc,1) && iy<=size(Fc,2)) if (Fc[ix,iy]>0.0) Plc[ix,iy] = 1.0; end; end
	if (ix<=size(Fv,1) && iy<=size(Fv,2)) if (Fv[ix,iy]>0.0) Plv[ix,iy] = 1.0; end; end

	return
end

@parallel_indices (ix,iy) function correct!(K::Data.Number, dt::Data.Number, C::Data.Number, cos_ϕ::Data.Number, sin_ϕ::Data.Number, sin_ψ::Data.Number, η_vp::Data.Number, λc::Data.Array, λv::Data.Array, Fc::Data.Array, Fv::Data.Array, Plc::Data.Array, Plv::Data.Array, τiic::Data.Array, τiiv::Data.Array, Ptc::Data.Array, Ptv::Data.Array, εxxc::Data.Array, εyyc::Data.Array, εxyv::Data.Array, εzzc::Data.Array, εxxv::Data.Array, εyyv::Data.Array, εxyc::Data.Array, εzzv::Data.Array, τxxc::Data.Array, τyyc::Data.Array, τxyv::Data.Array, τzzc::Data.Array, τxxv::Data.Array, τyyv::Data.Array, τxyc::Data.Array, τzzv::Data.Array, τxxc0::Data.Array, τyyc0::Data.Array, τzzc0::Data.Array, τxyv0::Data.Array, τxxv0::Data.Array, τyyv0::Data.Array, τzzv0::Data.Array, τxyc0::Data.Array, η_vec::Data.Array, η_vev::Data.Array, η_ec::Data.Array, η_ev::Data.Array, η_vepc::Data.Array)

	if (ix<=size(Plc,1) && iy<=size(Plc,2))  λc[ix,iy] = (Plc[ix,iy]*Fc[ix,iy])/(η_vec[ix,iy] + η_vp + K*dt*sin_ϕ*sin_ψ); end
	if (ix<=size(Plv,1) && iy<=size(Plv,2))  λv[ix,iy] = (Plv[ix,iy]*Fv[ix,iy])/(η_vev[ix,iy] + η_vp + K*dt*sin_ϕ*sin_ψ); end

	if (ix<=size(τxxc,1) && iy<=size(τxxc,2))  dQdτxxc = 0.5/τiic[ix,iy]*τxxc[ix,iy]; end # GPU style for non-optimal division
	if (ix<=size(τyyc,1) && iy<=size(τyyc,2))  dQdτyyc = 0.5/τiic[ix,iy]*τyyc[ix,iy]; end # GPU style for non-optimal division
	if (ix<=size(τzzc,1) && iy<=size(τzzc,2))  dQdτzzc = 0.5/τiic[ix,iy]*τzzc[ix,iy]; end # GPU style for non-optimal division
	if (ix<=size(τxyc,1) && iy<=size(τxyc,2))  dQdτxyc = 1.0/τiic[ix,iy]*τxyc[ix,iy]; end # GPU style for non-optimal division

	if (ix<=size(τxxv,1) && iy<=size(τxxv,2))  dQdτxxv = 0.5/τiiv[ix,iy]*τxxv[ix,iy]; end # GPU style for non-optimal division
	if (ix<=size(τyyv,1) && iy<=size(τyyv,2))  dQdτyyv = 0.5/τiiv[ix,iy]*τyyv[ix,iy]; end # GPU style for non-optimal division
	if (ix<=size(τzzv,1) && iy<=size(τzzv,2))  dQdτzzv = 0.5/τiiv[ix,iy]*τzzv[ix,iy]; end # GPU style for non-optimal division
	if (ix<=size(τxyv,1) && iy<=size(τxyv,2))  dQdτxyv = 1.0/τiiv[ix,iy]*τxyv[ix,iy]; end # GPU style for non-optimal division

	# Local effective strain-rate
	if (ix<=size(εxxc,1) && iy<=size(εxxc,2))  εxxc_eff = εxxc[ix,iy] + 0.5/η_ec[ix,iy]*τxxc0[ix,iy]; end
	if (ix<=size(εyyc,1) && iy<=size(εyyc,2))  εyyc_eff = εyyc[ix,iy] + 0.5/η_ec[ix,iy]*τyyc0[ix,iy]; end
	if (ix<=size(εzzc,1) && iy<=size(εzzc,2))  εzzc_eff = εzzc[ix,iy] + 0.5/η_ec[ix,iy]*τzzc0[ix,iy]; end
	if (ix<=size(εxyc,1) && iy<=size(εxyc,2))  εxyc_eff = εxyc[ix,iy] + 0.5/η_ec[ix,iy]*τxyc0[ix,iy]; end

	if (ix<=size(τxxc,1) && iy<=size(τxxc,2))  τxxc[ix,iy] = 2.0*η_vec[ix,iy]*(εxxc_eff -     λc[ix,iy]*dQdτxxc); end
	if (ix<=size(τyyc,1) && iy<=size(τyyc,2))  τyyc[ix,iy] = 2.0*η_vec[ix,iy]*(εyyc_eff -     λc[ix,iy]*dQdτyyc); end
	if (ix<=size(τzzc,1) && iy<=size(τzzc,2))  τzzc[ix,iy] = 2.0*η_vec[ix,iy]*(εzzc_eff -     λc[ix,iy]*dQdτzzc); end
	if (ix<=size(τxyc,1) && iy<=size(τxyc,2))  τxyc[ix,iy] = 2.0*η_vec[ix,iy]*(εxyc_eff - 0.5*λc[ix,iy]*dQdτxyc); end

	if (ix<=size(τxxv,1) && iy<=size(τxxv,2))  τxxv[ix,iy] = 2.0*η_vev[ix,iy]*(εxxv[ix,iy] + 0.5/η_ev[ix,iy]*τxxv0[ix,iy] -     λv[ix,iy]*dQdτxxv); end # GPU style for non-optimal division
	if (ix<=size(τyyv,1) && iy<=size(τyyv,2))  τyyv[ix,iy] = 2.0*η_vev[ix,iy]*(εyyv[ix,iy] + 0.5/η_ev[ix,iy]*τyyv0[ix,iy] -     λv[ix,iy]*dQdτyyv); end # GPU style for non-optimal division
	if (ix<=size(τzzv,1) && iy<=size(τzzv,2))  τzzv[ix,iy] = 2.0*η_vev[ix,iy]*(εzzv[ix,iy] + 0.5/η_ev[ix,iy]*τzzv0[ix,iy] -     λv[ix,iy]*dQdτzzv); end # GPU style for non-optimal division
	if (ix<=size(τxyv,1) && iy<=size(τxyv,2))  τxyv[ix,iy] = 2.0*η_vev[ix,iy]*(εxyv[ix,iy] + 0.5/η_ev[ix,iy]*τxyv0[ix,iy] - 0.5*λv[ix,iy]*dQdτxyv); end # GPU style for non-optimal division

	if (ix<=size(εxxc,1) && iy<=size(εxxc,2))  εiic_eff = sqrt( 0.5*( εxxc_eff*εxxc_eff + εyyc_eff*εyyc_eff + εzzc_eff*εzzc_eff ) + εxyc_eff*εxyc_eff ); end

	if (ix<=size(η_vepc,1) && iy<=size(η_vepc,2))  η_vepc[ix,iy] = (1.0-Plc[ix,iy])*η_vec[ix,iy] + Plc[ix,iy]*(0.5/εiic_eff*(C*cos_ϕ + Ptc[ix,iy]*sin_ϕ + η_vp*λc[ix,iy])); end

	return
end

@parallel function compute_R!(sin_ψ::Data.Number, dampX::Data.Number, dampY::Data.Number, dt::Data.Number, K::Data.Number, λc::Data.Array, RPt::Data.Array, Ptc::Data.Array, Ptc0::Data.Array, divVc::Data.Array, τxxc::Data.Array, τyyc::Data.Array, τxyv::Data.Array, Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array,
                              _dx::Data.Number, _dy::Data.Number)
	           # 0 = -div + divp + dive
			   # 0 = -div + λc*sin(ψ) + dive
			   # 0 = -div + λc*sin(ψ) - (Ptc1 - Ptc0)/K/dt
			   # 0 = -div + λc*sin(ψ) - (Ptc + K*dtl*amc*sin(ψ)  - Ptc0)/K/dt
			   # 0 = -div - (Ptc - Ptc0)/K/dt
	# @all(RPt)    = -@all(divVc)  + @all(λc)*sin(ψ) - 1.0/(K*dt)*(@all(Ptc1) - @all(Ptc0))
	# @all(RPt)    = -@all(divVc)  + @all(λc)*sin(ψ) - 1.0/(K*dt)*(@all(Ptc + K*dt*@all(λc)*sin(ψ)) - @all(Ptc0))
	@all(RPt)    = -@all(divVc)      - 1.0/(K*dt)*(@all(Ptc) - @all(Ptc0))
    @all(Rx)     = _dx*@d_xa(τxxc)    + _dy*@d_yxi(τxyv) - _dx*@d_xa(Ptc) - K*dt*sin_ψ*_dx*@d_xa(λc)
    @all(Ry)     = _dy*@d_ya(τyyc)    + _dx*@d_xyi(τxyv) - _dy*@d_ya(Ptc) - K*dt*sin_ψ*_dy*@d_ya(λc)
    @all(dVxdτ)  = dampX*@all(dVxdτ) + @all(Rx)
    @all(dVydτ)  = dampY*@all(dVydτ) + @all(Ry)

    return
end

@parallel_indices (ix,iy) function update_VP!(Vxe::Data.Array, Vye::Data.Array,  Ptc::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, RPt::Data.Array, dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array)

    if (ix<=size(Vxe,1)-2 && iy<=size(Vxe,2)-2) Vxe[ix+1,iy+1] = Vxe[ix+1,iy+1] + dτVx[ix,iy]*dVxdτ[ix,iy]; end # assumes Dirichlet E/W
    if (ix<=size(Vye,1)-2 && iy<=size(Vye,2)-2) Vye[ix+1,iy+1] = Vye[ix+1,iy+1] + dτVy[ix,iy]*dVydτ[ix,iy]; end # assumes Dirichlet N/S
	if (ix<=size(Ptc,1)   && iy<=size(Ptc,2)  ) Ptc[ix,iy]     = Ptc[ix,iy]     + dτPt[ix,iy]*RPt[ix,iy]; end

    return
end

@parallel_indices (ix,iy) function initialize!(xc, yc, xv, yv, Lx::Data.Number, Ly::Data.Number, rad::Data.Number, ε_bg::Data.Number, η_vec::Data.Array, η_vev::Data.Array, ηc::Data.Array, ηv::Data.Array, η_ec::Data.Array, η_ev::Data.Array, Vxe::Data.Array, Vye::Data.Array)

    if (ix<=size(xc, 1) && iy<=size(yc ,1)) radc2 = (xc[ix]-Lx*0.0)*(xc[ix]-Lx*0.0) + (yc[iy]-Ly*0.0)*(yc[iy]-Ly*0.0); end
    if (ix<=size(xv, 1) && iy<=size(yv ,1)) radv2 = (xv[ix]-Lx*0.0)*(xv[ix]-Lx*0.0) + (yv[iy]-Ly*0.0)*(yv[iy]-Ly*0.0); end

    if (ix<=size(η_ec,1) && iy<=size(η_ec,2)) if (radc2<rad*rad)  η_ec[ix,iy] = η_ec[ix,iy]/4.0; end; end
    if (ix<=size(η_ev,1) && iy<=size(η_ev,2)) if (radv2<rad*rad)  η_ev[ix,iy] = η_ev[ix,iy]/4.0; end; end

    if (ix<=size(η_vec,1) && iy<=size(η_vec,2)) η_vec[ix,iy] = 1.0/(1.0/ηc[ix,iy] + 1.0/η_ec[ix,iy]); end
    if (ix<=size(η_vev,1) && iy<=size(η_vev,2)) η_vev[ix,iy] = 1.0/(1.0/ηv[ix,iy] + 1.0/η_ev[ix,iy]); end

    if (ix<=size(Vxe,1)   && iy<=size(Vxe,2)-2) Vxe[ix,iy+1] = -ε_bg*(xv[ix]-Lx*0.0); end
    if (ix<=size(Vye,1)-2 && iy<=size(Vye,2)  ) Vye[ix+1,iy] =  ε_bg*(yv[iy]-Ly*0.0); end

    return
end

@parallel function update_Pt!(K::Data.Number, dt::Data.Number, sin_ψ::Data.Number, Ptc::Data.Array, λc::Data.Array)

	@all(Ptc)   = @all(Ptc)  +  K*dt*sin_ψ*@all(λc)

    return
end

# 2D Stokes routine
@views function Stokes2D_VEP()
# Physics
Lx        =  1.0
Ly        =  0.685
η0        =  2e10
dt        =  1e4
rad       =  5e-2
K         =  2.0
G0        =  1.0
ε_bg      =  5e-6/dt
C         =  1.75e-4
ϕ         =  30*π/180
ψ         =  10*π/180
η_vp      =  1*2.5e2
time_p    =  0.0
# Numerics
nt        =  20
nitr 	  =  5e4
nout      =  1000
BLOCK_X   =  16
BLOCK_Y   =  16
GRID_X    =  6
GRID_Y    =  4
Vdmp      =  5.0
scV       =  2.0
scPt      =  5.0
# β_n       =  2.0 
ε_nl      =  1e-11
# Preprocessing
nx        = GRID_X*BLOCK_X - 2 # -2 due to overlength of array nx+2
ny        = GRID_Y*BLOCK_Y - 2 # -2 due to overlength of array ny+2
cuthreads = (BLOCK_X, BLOCK_Y, 1)
cublocks  = (GRID_X , GRID_Y , 1)
dx        = Lx/nx
dy        = Ly/ny
_dx       = 1.0/dx
_dy       = 1.0/dy
# Initialisation
Ptc       = @zeros(nx  ,ny  )
Ptv       = @zeros(nx+1,ny+1)
Ptc0      = @zeros(nx  ,ny  )
Ptv0      = @zeros(nx+1,ny+1)
RPt       = @zeros(nx  ,ny  )
dτPt      = @zeros(nx  ,ny  )
divVc     = @zeros(nx  ,ny  )
Vxe       = @zeros(nx+1,ny+2)
Vye       = @zeros(nx+2,ny+1)
εxxc      = @zeros(nx  ,ny  )
εxxv      = @zeros(nx+1,ny+1)
εyyc      = @zeros(nx  ,ny  )
εyyv      = @zeros(nx+1,ny+1)
εzzc      = @zeros(nx  ,ny  )
εzzv      = @zeros(nx+1,ny+1)
εxyc      = @zeros(nx  ,ny  )
εxyv      = @zeros(nx+1,ny+1)
τxxc      = @zeros(nx  ,ny  )
τxxv      = @zeros(nx+1,ny+1)
τyyc      = @zeros(nx  ,ny  )
τyyv      = @zeros(nx+1,ny+1)
τzzc      = @zeros(nx  ,ny  )
τzzv      = @zeros(nx+1,ny+1)
τxyc      = @zeros(nx  ,ny  )
τxyv      = @zeros(nx+1,ny+1)
τxxc0     = @zeros(nx  ,ny  )
τxxv0     = @zeros(nx+1,ny+1)
τyyc0     = @zeros(nx  ,ny  )
τyyv0     = @zeros(nx+1,ny+1)
τzzc0     = @zeros(nx  ,ny  )
τzzv0     = @zeros(nx+1,ny+1)
τxyc0     = @zeros(nx  ,ny  )
τxyv0     = @zeros(nx+1,ny+1)
τiic      = @zeros(nx  ,ny  )
τiiv      = @zeros(nx+1,ny+1)
Rx        = @zeros(nx-1,ny  )
Ry        = @zeros(nx  ,ny-1)
dVxdτ     = @zeros(nx-1,ny  )
dVydτ     = @zeros(nx  ,ny-1)
dτVx      = @zeros(nx-1,ny  )
dτVy      = @zeros(nx  ,ny-1)
Fv        = @zeros(nx+1,ny+1)
Fc        = @zeros(nx  ,ny  )
Plv       = @zeros(nx+1,ny+1)
Plc       = @zeros(nx  ,ny  )
λv        = @zeros(nx+1,ny+1)
λc        = @zeros(nx  ,ny  )
ηv        =    η0*@ones(nx+1,ny+1)
ηc        =    η0*@ones(nx  ,ny  )
η_ev      = G0*dt*@ones(nx+1,ny+1)
η_ec      = G0*dt*@ones(nx  ,ny  )
η_vev     =       @ones(nx+1,ny+1)
η_vec     =       @ones(nx  ,ny  )
η_vepc    =       @ones(nx  ,ny  )
# Weights vertices
wSW       = @ones(nx+1,ny+1)
wSE       = @ones(nx+1,ny+1)
wNW       = @ones(nx+1,ny+1)
wNE       = @ones(nx+1,ny+1)
# Dummy avg tables vertices
AvSW      = @zeros(nx+1,ny+1)
AvSE      = @zeros(nx+1,ny+1)
AvNW      = @zeros(nx+1,ny+1)
AvNE      = @zeros(nx+1,ny+1)
# Init coord
xc        = LinRange(dx/2, Lx-dx/2, nx  )
yc        = LinRange(dy/2, Ly-dy/2, ny  )
xv        = LinRange(0.0 , Lx     , nx+1)
yv        = LinRange(0.0 , Ly     , ny+1)
min_dxy2  = min(dx,dy)^2
max_nxy   = max(nx,ny)
dampX     = (1.0-Vdmp/nx)
dampY     = (1.0-Vdmp/ny)
sin_ψ     = sin(ψ)
sin_ϕ     = sin(ϕ)
cos_ϕ     = cos(ϕ)
@printf("dt = %2.3e\n", dt)
# action
@parallel cublocks cuthreads weights!(wSW, wSE, wNW, wNE)

@parallel cublocks cuthreads initialize!(xc, yc, xv, yv, Lx, Ly, rad, ε_bg, η_vec, η_vev, ηc, ηv, η_ec, η_ev, Vxe, Vye)

tim_evo=[]; τii_max=[];
for it = 1:nt

 	@parallel cublocks cuthreads swap0ld!(τxxc, τxxc0, τyyc, τyyc0, τzzc, τzzc0, τxyv, τxyv0, Ptc, Ptc0, τxxv, τxxv0, τyyv, τyyv0, τzzv, τzzv0, τxyc, τxyc0, Ptv, Ptv0)

 	global err=2*ε_nl; global err_evo1=[]; global err_evo2=[];

 	for itr = 1:nitr

 		@parallel cublocks cuthreads reset!(λc, λv)

		@parallel cublocks cuthreads setBCvel!(Vxe, Vye)

		@parallel cublocks cuthreads compute_ε!(Vxe, Vye, εxxc, εyyc, εxyv, εzzc, divVc, _dx, _dy)

		@parallel cublocks cuthreads c2v!(εxxv, εxxc, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
		@parallel cublocks cuthreads c2v!(εyyv, εyyc, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
		@parallel cublocks cuthreads c2v!(εzzv, εzzc, AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
		@parallel cublocks cuthreads c2v!(Ptv,  Ptc,  AvSW, AvSE, AvNW, AvNE, wSW, wSE, wNW, wNE)
		@parallel cublocks cuthreads v2c!(εxyc, εxyv)

		@parallel cublocks cuthreads compute_τ!(εxxc, εyyc, εxyv, εzzc, εxxv, εyyv, εxyc, εzzv, τxxc, τyyc, τxyv, τzzc, τxxv, τyyv, τxyc, τzzv, τxxc0, τyyc0, τzzc0, τxyv0, τxxv0, τyyv0, τzzv0, τxyc0, η_vec, η_vev, η_ec, η_ev)

		@parallel cublocks cuthreads check_yield!(K, dt, sin_ψ, sin_ϕ, cos_ϕ, C, η_vp, Ptc, Ptv, λc, λv, Fc, Fv, τiic, τiiv, τxxc, τyyc, τzzc, τxyc, τxxv, τyyv, τzzv, τxyv)
		
		if mod(itr, nout)==0
			max_Fc  = maximum(Array(Fc[:]))
			max_Fv  = maximum(Array(Fv[:]))
			@printf("Fc = %1.3e \n", max_Fc)
			@printf("Fv = %1.3e \n", max_Fv)
		end

		@parallel cublocks cuthreads plastic_nodes!(Fc, Fv, Plc, Plv)

		if mod(itr, nout)==0
			max_Plc = sum(Array(Plc[:]))
			max_Plv = sum(Array(Plv[:]))
			@printf("Number of plastic centres  = %1d \n", max_Plc)
			@printf("Number of plastic vertices = %1d \n", max_Plv)
		end

		@parallel cublocks cuthreads correct!(K, dt, C, cos_ϕ, sin_ϕ, sin_ψ, η_vp, λc, λv, Fc, Fv, Plc, Plv, τiic, τiiv, Ptc, Ptv, εxxc, εyyc, εxyv, εzzc, εxxv, εyyv, εxyc, εzzv, τxxc, τyyc, τxyv, τzzc, τxxv, τyyv, τxyc, τzzv, τxxc0, τyyc0, τzzc0, τxyv0, τxxv0, τyyv0, τzzv0, τxyc0, η_vec, η_vev, η_ec, η_ev, η_vepc)

		@parallel cublocks cuthreads check_yield!(K, dt, sin_ψ, sin_ϕ, cos_ϕ, C, η_vp, Ptc, Ptv, λc, λv, Fc, Fv, τiic, τiiv, τxxc, τyyc, τzzc, τxyc, τxxv, τyyv, τzzv, τxyv)
		
		if mod(itr, nout)==0
			max_Fc  = maximum(Array(Fc[:]))
			max_Fv  = maximum(Array(Fv[:]))
			@printf("Check Fc = %1.3e \n", max_Fc)
			@printf("Check Fv = %1.3e \n", max_Fv)
		end

		@parallel cublocks cuthreads timesteps!(scPt, scV, min_dxy2, max_nxy, dτVx, dτVy, dτPt, η_vec)

		@parallel cublocks cuthreads compute_R!(sin_ψ, dampX, dampY, dt, K, λc, RPt, Ptc, Ptc0, divVc, τxxc, τyyc, τxyv, Rx, Ry, dVxdτ, dVydτ, _dx, _dy);

		@parallel cublocks cuthreads update_VP!(Vxe, Vye, Ptc, dVxdτ, dVydτ, RPt, dτVx, dτVy, dτPt);

		# convergence check
		if mod(itr,nout)==0
			global norm_Rx, norm_Ry, norm_RPt
			norm_Rx   = norm(Array(Rx[:]))/length(Array(Rx[:]))
			norm_Ry   = norm(Array(Ry[:]))/length(Array(Ry[:]))
			norm_RPt  = norm(Array(RPt[:]))/length(Array(RPt[:]))
			err = maximum([norm_Rx, norm_Ry, norm_RPt])
			push!(err_evo1,maximum([norm_Rx, norm_Ry, norm_RPt])); push!(err_evo2,itr);
			@printf("\n> it %d, iter %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPt=%1.3e] \n", it, itc, err, norm_Rx, norm_Ry, norm_RPt)
		end
		if (err <= ε_nl) break; end
		global itc=itr
 	end
	# Include plastic correction in converge pressure
	@parallel cublocks cuthreads update_Pt!(K, dt, sin_ψ, Ptc, λc)
	time_p = time_p + dt
	# record loading
	push!(tim_evo,time_p); push!(τii_max, mean(Array(τiic[:])));
	p1 = plot(tim_evo, τii_max, legend=false, xlabel="# steps", ylabel="mean(τiic)", linewidth=2, markershape=:circle, markersize=3)
	p2 = heatmap(xc,yc,transpose(Array(τiic)), aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(dy/2, Ly-dy/2), c=:viridis, title="τiic")
	# p2 = heatmap(xv,yc,transpose(Vx), aspect_ratio=1, xlims=(0, Lx), ylims=(dy/2, Ly-dy/2), c=:viridis, title="Vx") #clims=((0.0002, 0.00059))
	display(plot( p2 ))
end

epsilon = "1e_11"
run_id  = "v6"
save_array(Vxe[:,2:end-1] , string("Vx_", epsilon, "_", run_id), nt)
save_array(Vye[2:end-1,:] , string("Vy_", epsilon, "_", run_id), nt)
save_array(Ptc , string("Ptc_", epsilon, "_", run_id), nt)
save_array(τiic, string("Tiic_", epsilon, "_", run_id), nt)
save_array(η_ec, string("etaec_", epsilon, "_", run_id), nt)

# Plotting
# p1 = heatmap(xc,yc,transpose(Pt), aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(dy/2, Ly-dy/2), c=:inferno, title="Pressure");
# p2 = heatmap(xc,yv,transpose(Vy), aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="Vy");
# p4 = heatmap(xc,yv[2:end-1],transpose(log10.(abs.(dVydτ))), aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(dy, Ly-dy), c=:inferno, title="log10(Ry)");
# p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10 )
# display(plot( p1, p2, p4, p5 ))

# A_eff = (3*2)/1e9*nx*ny*sizeof(Data.Number);     # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
# t_it  = time_s/(itc-warmup);             # Execution time per iteration [s]
# T_eff = A_eff/t_it;                      # Effective memory throughput [GB/s]
# @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ %1.2f GB/s) \n", itc, err, time_s, T_eff)

return nothing
end

Stokes2D_VEP()
