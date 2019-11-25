using Printf
#################################################

if USE_GPU
	ENV["CUDA_VISIBLE_DEVICES"]=ENV["SLURM_LOCALID"]
	using CUDAdrv, CUDAnative, CuArrays
	myzeros(dims...) = cuzeros(DAT, dims...);
	myones(dims...)  = cuones(DAT, dims...);
	myrand(dims...)  = CuArray(rand(DAT, dims...));
	DatArray         = CuArray{DAT,3}
	DatArray_k       = CuDeviceArray{DAT,3} # to be used inside kernels
	using CUDAnative: log, exp, tanh, pow # TODO: works only as long as these are not used outside of kernels...
else
	myzeros(dims...) = zeros(DAT, dims...);
	myones(dims...)  = ones(DAT, dims...);
	myrand(dims...)  = rand(DAT, dims...);
	DatArray         = Array{DAT,3}
	DatArray_k       = Array{DAT,3}
	pow(x,y)         = x^y
end

#######################

#TODO: add!
if USE_MPI
	push!(LOAD_PATH, "./gg.jl/src") # ATTENTION: the GG package needs to be present in this subfolder - later installed normally via package manager.
	using GG
	import MPI
	#TODO: these definitions are just temporarily here; they can go into the GG package. Also nprocs does not have to be get from MPI everytime if perf relevant.
	maximum_g(A) = (max_l  = maximum(A); MPI.Allreduce(max_l,  MPI.MAX, MPI.COMM_WORLD))
	minimum_g(A) = (min_l  = minimum(A); MPI.Allreduce(min_l,  MPI.MIN, MPI.COMM_WORLD))
	mean_g(A)    = (mean_l = mean(A);    MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD)/MPI.Comm_size(MPI.COMM_WORLD))
else
	maximum_g(A) = maximum(A)
	minimum_g(A) = minimum(A)
	mean_g(A)    = mean(A)
end

#################################################
## Diverse functions
@views function diff2(A, dim)                                                          #SO: in Julia, diff does not seem to exist for 3D arrays.
	if     (dim==1); (A[3:end,:,:]-A[2:end-1,:,:]) .- (A[2:end-1,:,:]-A[1:end-2,:,:])
	elseif (dim==2); (A[:,3:end,:]-A[:,2:end-1,:]) .- (A[:,2:end-1,:]-A[:,1:end-2,:])
	elseif (dim==3); (A[:,:,3:end]-A[:,:,2:end-1]) .- (A[:,:,2:end-1]-A[:,:,1:end-2])
	end
end

@views all(A)      = A[1:end,1:end,1:end]
@views inn(A)      = A[2:end-1,2:end-1,2:end-1]
@views inn_av_z(A) = (A[2:end-1,2:end-1,2:end-2] .+ A[2:end-1,2:end-1,3:end-1]).*0.5;
@views in_x(A)     = A[2:end-1,:,:]
@views in_y(A)     = A[:,2:end-1,:]
@views in_z(A)     = A[:,:,2:end-1]
@views bc_no_dx(A) = ( A[1,:,:] .= A[2,:,:]; A[end,:,:] .= A[end-1,:,:]; ) #SO: TODO: not sure if without return nothing it will return the last statement even if it is not caught? Test later...
@views bc_no_dy(A) = ( A[:,1,:] .= A[:,2,:]; A[:,end,:] .= A[:,end-1,:]; )
@views bc_no_dz(A) = ( A[:,:,1] .= A[:,:,2]; A[:,:,end] .= A[:,:,end-1]; )
@views bc_z_val(A,val) = ( A[:,:,1] .= val; A[:,:,end] .= val; )
@views bc_dz_period(A) = ( A[:,:,1] .= A[:,:,end-1]; A[:,:,end] .= A[:,:,2]; )
detrend(A)         = A .- mean_g(A)                                               #SO: A .- mean(A) is the simplest implementation according to the MATLAB function detrend, but it should rather remove the linear trend, as I remember, right? So, it should use the derivatives, I guess.
## LR: save funtion
function SaveArray(Aname, A, isave)
	fname = string(Aname, "_", isave, ".bin")
	out = open(fname,"w"); write(out,A); close(out)
end



#################################################
## Macros to enable a single code for CPU and GPU
macro threadids_or_loop(sizemax, block)
    @static if USE_GPU
    	esc(
    	    quote
        		ix = (blockIdx().x-1) * blockDim().x + threadIdx().x # thread ID, dimension x
        		iy = (blockIdx().y-1) * blockDim().y + threadIdx().y # thread ID, dimension y
        		iz = (blockIdx().z-1) * blockDim().z + threadIdx().z # thread ID, dimension z
        		$block
    	    end
    	)
    else
    	esc(
            quote
        		@threads for iz=1:$sizemax[3]
        			for iy=1:$sizemax[2], ix=1:$sizemax[1]
        		    	$block
        			end
        		end
            end
    	)
    end
end

macro kernel(cublocks, cuthreads, kernel)
    @static if USE_GPU
	    esc(:( @cuda blocks=$cublocks threads=$cuthreads $kernel ))
    else
        esc(:( $kernel ))
    end
end

macro devicesync()
    @static if USE_GPU
	    esc(:( CUDAdrv.synchronize() ))
    end
end

##############################
## Macros from cuda_scientific
args(A) = esc.((A,:ix,:iy,:iz,:ixi,:iyi,:izi))
macro   d_xa(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix+1,$iy  ,$iz  ] - $A[$ix  ,$iy  ,$iz  ] ) end
macro   d_ya(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix  ,$iy+1,$iz  ] - $A[$ix  ,$iy  ,$iz  ] ) end
macro   d_za(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix  ,$iy  ,$iz+1] - $A[$ix  ,$iy  ,$iz  ] ) end
macro   d_xi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix+1,$iyi ,$izi ] - $A[$ix  ,$iyi ,$izi ] ) end
macro   d_yi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ixi ,$iy+1,$izi ] - $A[$ixi ,$iy  ,$izi ] ) end
macro   d_zi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ixi ,$iyi ,$iz+1] - $A[$ixi ,$iyi ,$iz  ] ) end
macro    all(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix  ,$iy  ,$iz  ] ) end   #TODO: see if this needs esc or what is wrong, why it fails in line 120, saying k_mufi is not defined.
macro    inn(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ixi ,$iyi ,$izi ] ) end
macro   in_x(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ixi ,$iy  ,$iz  ] ) end
macro   in_y(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix  ,$iyi ,$iz  ] ) end
macro   in_z(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix  ,$iy  ,$izi ] ) end
macro  in_yz(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ix  ,$iyi ,$izi ] ) end
macro  in_xz(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ixi ,$iy  ,$izi ] ) end
macro  in_xy(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :( $A[$ixi ,$iyi ,$iz  ] ) end
macro av_xyi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$izi ] +
                                                     $A[$ix+1,$iy  ,$izi ] +
                                                     $A[$ix  ,$iy+1,$izi ] +
                                                     $A[$ix+1,$iy+1,$izi ] )*0.25) end
macro av_xzi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iyi ,$iz  ] +
                                                     $A[$ix+1,$iyi ,$iz  ] +
                                                     $A[$ix  ,$iyi ,$iz+1] +
                                                     $A[$ix+1,$iyi ,$iz+1] )*0.25) end
macro av_yzi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ixi ,$iy  ,$iz  ] +
                                                     $A[$ixi ,$iy+1,$iz  ] +
                                                     $A[$ixi ,$iy  ,$iz+1] +
                                                     $A[$ixi ,$iy+1,$iz+1] )*0.25) end
macro av_xya(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ] +
                                                     $A[$ix+1,$iy  ,$iz  ] +
                                                     $A[$ix  ,$iy+1,$iz  ] +
                                                     $A[$ix+1,$iy+1,$iz  ] )*0.25) end
macro av_xza(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ] +
                                                     $A[$ix+1,$iy  ,$iz  ] +
                                                     $A[$ix  ,$iy  ,$iz+1] +
                                                     $A[$ix+1,$iy  ,$iz+1] )*0.25) end
macro av_yza(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ] +
                                                     $A[$ix  ,$iy+1,$iz  ] +
                                                     $A[$ix  ,$iy  ,$iz+1] +
                                                     $A[$ix  ,$iy+1,$iz+1] )*0.25) end
macro  av_xa(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ] +
                                                     $A[$ix+1,$iy  ,$iz  ] )*0.5) end
macro  av_ya(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ] +
                                                     $A[$ix  ,$iy+1,$iz  ] )*0.5) end
macro  av_za(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ] +
                                                     $A[$ix  ,$iy  ,$iz+1] )*0.5) end
macro  av_xi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iyi ,$izi ] +
                                                     $A[$ix+1,$iyi ,$izi ] )*0.5) end
macro  av_yi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ixi ,$iy  ,$izi ] +
                                                     $A[$ixi ,$iy+1,$izi ] )*0.5) end
macro  av_zi(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ixi ,$iyi ,$iz  ] +
                                                     $A[$ixi ,$iyi ,$iz+1] )*0.5) end

# Anton's invariant definition (not cancelling the closest neighbours - averaging the squared values)
macro av_xya2(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ]*$A[$ix  ,$iy  ,$iz  ] +
                                                      $A[$ix+1,$iy  ,$iz  ]*$A[$ix+1,$iy  ,$iz  ] +
                                                      $A[$ix  ,$iy+1,$iz  ]*$A[$ix  ,$iy+1,$iz  ] +
                                                      $A[$ix+1,$iy+1,$iz  ]*$A[$ix+1,$iy+1,$iz  ] )*0.25) end
macro av_xza2(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ]*$A[$ix  ,$iy  ,$iz  ] +
                                                      $A[$ix+1,$iy  ,$iz  ]*$A[$ix+1,$iy  ,$iz  ] +
                                                      $A[$ix  ,$iy  ,$iz+1]*$A[$ix  ,$iy  ,$iz+1] +
                                                      $A[$ix+1,$iy  ,$iz+1]*$A[$ix+1,$iy  ,$iz+1] )*0.25) end
macro av_yza2(A)  A,ix,iy,iz,ixi,iyi,izi=args(A); :(( $A[$ix  ,$iy  ,$iz  ]*$A[$ix  ,$iy  ,$iz  ] +
                                                      $A[$ix  ,$iy+1,$iz  ]*$A[$ix  ,$iy+1,$iz  ] +
                                                      $A[$ix  ,$iy  ,$iz+1]*$A[$ix  ,$iy  ,$iz+1] +
                                                      $A[$ix  ,$iy+1,$iz+1]*$A[$ix  ,$iy+1,$iz+1] )*0.25) end

macro maxloc(A)  esc(:( max( max( max( max($A[ixi-1,iyi  ,izi  ], $A[ixi+1,iyi  ,izi  ])  , $A[ixi  ,iyi  ,izi  ] ),
                                       max($A[ixi  ,iyi-1,izi  ], $A[ixi  ,iyi+1,izi  ]) ),
                                       max($A[ixi  ,iyi  ,izi-1], $A[ixi  ,iyi  ,izi+1]) ) )) end

#TODO: later, participate should all be shifted for potential better performance (probably not easily relevant). Requires though adaption of the functions d_xa etc.
macro participate_a(A)   A,ix,iy,iz,ixi,iyi,izi=args(A);  :($ix<=size($A,1)   && $iy<=size($A,2)   && $iz<=size($A,3)  ) end
macro participate_i(A)   A,ix,iy,iz,ixi,iyi,izi=args(A);  :($ix<=size($A,1)-2 && $iy<=size($A,2)-2 && $iz<=size($A,3)-2) end
macro participate_ix(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :($ix<=size($A,1)-2 && $iy<=size($A,2)   && $iz<=size($A,3)  ) end
macro participate_iy(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :($ix<=size($A,1)   && $iy<=size($A,2)-2 && $iz<=size($A,3)  ) end
macro participate_iz(A)  A,ix,iy,iz,ixi,iyi,izi=args(A);  :($ix<=size($A,1)   && $iy<=size($A,2)   && $iz<=size($A,3)-2) end


function Centroid2VerticesOnCPU!( fv, fc )
	@. fv                          = 0.0
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,1:end-1,1:end-1]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,1:end-1,1:end-1]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,2:end-0,1:end-1]
    @. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,1:end-1,2:end-0]

	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,2:end-0,2:end-0]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[1:end-1,2:end-0,2:end-0]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,1:end-1,2:end-0]
	@. fv[2:end-1,2:end-1,2:end-1] += 1.0/8.0 * fc[2:end-0,2:end-0,1:end-1]
end

function ExCentroid2VerticesOnCPU!( fv, fc_ex )
	@. fv = 0.0
	@. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,1:end-1]
    @. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,2:end-0]

	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,1:end-1]
end

function Vertices2VyOnCPU!( fvy, fv )
	@. fvy = 0
	@. fvy += 1.0/4.0 * fv[1:end-1,:,1:end-1]
	@. fvy += 1.0/4.0 * fv[2:end-0,:,2:end-0]
	@. fvy += 1.0/4.0 * fv[1:end-1,:,2:end-0]
	@. fvy += 1.0/4.0 * fv[2:end-0,:,1:end-1]
end

function ExCentroid2VyOnCPU!( Ty, fc_ex )

	nx,ny,nz=size(fc_ex,1),size(fc_ex,2),size(fc_ex,3)
	fv  = myzeros(nx-1,ny-1,nz-1)
	@. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,1:end-1]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,1:end-1]
    @. fv += 1.0/8.0 * fc_ex[1:end-1,1:end-1,2:end-0]

	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[1:end-1,2:end-0,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,1:end-1,2:end-0]
	@. fv += 1.0/8.0 * fc_ex[2:end-0,2:end-0,1:end-1]

	@. Ty = 0
	@. Ty += 1.0/4.0 * fv[1:end-1,:,1:end-1]
	@. Ty += 1.0/4.0 * fv[2:end-0,:,2:end-0]
	@. Ty += 1.0/4.0 * fv[1:end-1,:,2:end-0]
	@. Ty += 1.0/4.0 * fv[2:end-0,:,1:end-1]

	@printf("Interpolation done! min(fvy) = %f - max(fvy) = %f\n", minimum(Ty), maximum(Ty) )

end
