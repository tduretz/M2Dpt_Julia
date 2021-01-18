# This script is benchmarked against Taras code - best advection

# Initialisation
using Printf
using Statistics
using Plots
using LoopVectorization
using MAT
# pyplot(size = (750/1, 565/1))

path = "/Users/imac/REPO_GIT/M2Dpt_Julia/Markers2D/"

gr(size = (750/1, 565/1))

########################################################################
mutable struct Markers
    x         ::  Array{Float64,1}
    y         ::  Array{Float64,1}
    Vx        ::  Array{Float64,1}
    Vy        ::  Array{Float64,1}
    phase     ::  Array{Int64,1}
end
########################################################################
# Interpolation from Vx nodes to particles
@views function VxFromVxNodes(Vx, k, p, xv, yce, dx, dy, ncx, ncy, new)
    # Interpolate vx
    i = Int64(round(trunc( (p.x[k] -  xv[1])/dx ) + 1));
    j = Int64(round(trunc( (p.y[k] - yce[1])/dy ) + 1));
    if i<1
        i = 1
    elseif i>ncx
        i = ncx
    end
    if j<1
        j = 1;
    elseif j> ncy+1
        j = ncy+1
    end
    # Compute distances
    dxmj = p.x[k] -  xv[i]
    dymi = p.y[k] - yce[j]
    # Compute vx velocity for the top and bottom of the cell
    vxm13 = Vx[i,j  ] * (1-dxmj/dx) + Vx[i+1,j  ]*dxmj/dx
    vxm24 = Vx[i,j+1] * (1-dxmj/dx) + Vx[i+1,j+1]*dxmj/dx
    if new==1 
        if dxmj/dx>=0.5
            if i<ncx
                vxm13 += 0.5*((dxmj/dx-0.5)^2) * (Vx[i,j  ] - 2.0*Vx[i+1,j  ] + Vx[i+2,j  ]);
                vxm24 += 0.5*((dxmj/dx-0.5)^2) * (Vx[i,j+1] - 2.0*Vx[i+1,j+1] + Vx[i+2,j+1]);
            end
        else
            if i>1
                vxm13 += 0.5*((dxmj/dx-0.5)^2) * (Vx[i-1,j  ] - 2.0*Vx[i,j  ] + Vx[i+1,j  ]);
                vxm24 += 0.5*((dxmj/dx-0.5)^2) * (Vx[i-1,j+1] - 2.0*Vx[i,j+1] + Vx[i+1,j+1]);
            end
        end
    end
    # Compute vx
    vxm = (1-dymi/dy) * vxm13 + (dymi/dy) * vxm24
    return vxm
end
@views function VyFromVxNodes(Vy, k, p, xce, yv, dx, dy, ncx, ncy, new)
    # Interpolate vy
    i = Int64(round(trunc( (p.x[k] - xce[1])/dx ) + 1));
    j = Int64(round(trunc( (p.y[k] -  yv[1])/dy ) + 1));
    if i<1
        i=1
    elseif i>ncx+1
        i=ncx+1
    end
    if j<1
        j=1
    elseif j>ncy
        j = ncy
    end
    # Compute distances
    dxmj = p.x[k] - xce[i]
    dymi = p.y[k] -  yv[j]
    # Compute vy velocity for the left and right of the cell
    vym12 = Vy[i,j  ]*(1-dymi/dy) + Vy[i  ,j+1]*dymi/dy
    vym34 = Vy[i+1,j]*(1-dymi/dy) + Vy[i+1,j+1]*dymi/dy
    if new==1 
        if dymi/dy>=0.5
            if j<ncy
                vym12 += 0.5*((dymi/dy-0.5)^2) * ( Vy[i,j  ] - 2.0*Vy[i,j+1  ] + Vy[i,j+2  ]);
                vym34 += 0.5*((dymi/dy-0.5)^2) * ( Vy[i+1,j] - 2.0*Vy[i+1,j+1] + Vy[i+1,j+2]);
            end      
        else
            if j>1
                vym12 += 0.5*((dymi/dy-0.5)^2) * ( Vy[i,j-1  ] - 2.0*Vy[i,j  ] + Vy[i,j+1  ]);
                vym34 += 0.5*((dymi/dy-0.5)^2) * ( Vy[i+1,j-1] - 2.0*Vy[i+1,j] + Vy[i+1,j+1]);
            end
        end
    end
    # Compute vy
    vym = (1-dxmj/dx)*vym12 + (dxmj/dx)*vym34
    return vym
end
########################################################################
function Markers2D()
@printf("Running on %d thread(s)\n", Threads.nthreads())
# Import data
file = matopen( path * "data41_benchmark.mat")
Vx   = read(file, "Vx")  # ACHTUNG THIS CONTAINS GHOST NODES AT NORTH/SOUTH
Vy   = read(file, "Vy")  # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
Pt   = read(file, "Pt")
dt   = read(file, "dt")  # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
nt   = Int64(read(file, "ntimesteps")) # Number of steps
xmi  = read(file, "xm0")               # initial positions
ymi  = read(file, "ym0")
xmf  = read(file, "xm4")               # final positions
ymf  = read(file, "ym4")
close(file)
########
itpw  = 1.0/3.0
Vizu  = 0
C     = 0.25
nout  = 10
xmin  = 0
xmax  = 100
ymin  = 0
ymax  = 100
ncx   = 40
ncy   = 40
nmark = length(xmi) # total initial number of marker in grid
# Spacing
dx, dy  = (xmax-xmin)/ncx, (ymax-ymin)/ncy
# 1D coordinates 
xc          = LinRange(xmin+dx/2, xmax-dx/2, ncx)
xce         = LinRange(xmin-dx/2, xmax+dx/2, ncx+2)
yc          = LinRange(ymin+dy/2, ymax-dy/2, ncy)
yce         = LinRange(xmin-dy/2, xmax+dy/2, ncy+2)
xv          = LinRange(xmin, xmax, ncx+1)
yv          = LinRange(ymin, ymax, ncy+1)
# 2D mesh
(xc2,yc2)   = ([x for x=xc,y=yc], [y for x=xc,y=yc]);
(xv2,yv2)   = ([x for x=xv,y=yv], [y for x=xv,y=yv]);
(xvx2,yvx2) = ([x for x=xv,y=yc], [y for x=xv,y=yc]);
(xvy2,yvy2) = ([x for x=xc,y=yv], [y for x=xc,y=yv]);
# 2D tables
mpc         = zeros(Float64,(ncx  ,ncy  )) # markers per cell
phase       = zeros(Float64,(ncx  ,ncy  ))
@printf( "dx = %2.2e --- dy = %2.2e --- dt = %2.2e\n", dx, dy, dt )
# Initialise markers
xm   = vec(xmi)
ym   = vec(ymi)
Vxm  =-vec(ymi)
Vym  = vec(xmi)
phm  = zeros(Int64,   size(xm))
p    = Markers( xm, ym, Vxm, Vym, phm )
# define phase
for k=1:nmark
    if (p.x[k]<p.y[k]) 
        p.phase[k] = 1
    end
end

# RK4 weights
rkw = 1.0/6.0*[1.0 2.0 2.0 1.0] # for averaging
rkv = 1.0/2.0*[1.0 1.0 2.0 2.0] # for time stepping

# Time loop
for it=1:nt
    
    # BASIC FUNCTION: Count number of marker per cell
    @avx for j=1:ncy, i=1:ncx
        mpc[i,j] = 0.0
    end
    @simd for k=1:nmark # @avx ne marche pas ici
        # Get the column:
        dstx = p.x[k] - xc[1];
        i    = Int64(round(ceil( (dstx/dx) + 0.5)));
        # Get the line:
        dsty = p.y[k] - yc[1];
        j   = Int64(round(ceil( (dsty/dy) + 0.5)));
        # Increment cell count
        @inbounds mpc[i,j] += 1.0
    end

    # BASIC FUNCTION: Interpolate field from markers to centers
    weight         = zeros(Float64,(ncx  ,ncy  )) 
    @avx for j=1:ncy, i=1:ncx
        phase[i,j] = 0
    end 
    @simd for k=1:nmark # @avx ne marche pas ici
        if (p.phase[k]>=0)
            # Get the column:
            dstx = p.x[k] - xc[1];
            i    = Int(ceil( (dstx/dx) + 0.5));
            # Get the line:
            dsty = p.y[k] - yc[1];
            j    = Int(ceil( (dsty/dy) + 0.5));
            # Relative distances
            dxm = 2.0*abs( xc[i] - p.x[k]);
            dym = 2.0*abs( yc[j] - p.y[k]);
            # Increment cell count
            @inbounds phase[i,j]  += p.phase[k] * (1-dxm/dx)*(1-dym/dy);
            @inbounds weight[i,j] +=              (1-dxm/dx)*(1-dym/dy);
        end
    end
    @avx for j=1:ncy, i=1:ncx
       phase[i,j] /= weight[i,j]
    end 
    
    # BASIC FUNCTION: Marker advection with 4th order Roger-Gunther
    @simd for k=1:nmark
        if (p.phase[k]>=0)
            x0 = p.x[k];
            y0 = p.y[k];
            vx = 0.0
            vy = 0.0
            # Roger-Gunther loop
            for rk=1:4
                # Interp velocity from grid
                vxm = VxFromVxNodes(Vx, k, p, xv, yce, dx, dy, ncx, ncy, 1)
                vym = VyFromVxNodes(Vy, k, p, xce, yv, dx, dy, ncx, ncy, 1)
                # Temporary RK advection steps
                p.x[k] = x0 + rkv[rk]*dt*vxm
                p.y[k] = y0 + rkv[rk]*dt*vym
                # Average final velocity 
                vx    += rkw[rk]*vxm
                vy    += rkw[rk]*vym
            end
            # Advect points
            p.x[k] = x0 + rkv[4]*dt*vx
            p.y[k] = y0 + rkv[4]*dt*vy
        end
    end
end

@printf("Difference with Taras: %2.2e\n", mean(xmf[:].-p.x[:]) )

it = nt
if (Vizu == 1 && (it==1 || mod(it,nout)==0) )
    p1 = heatmap(xc, yc, transpose(mpc),c=:inferno,aspect_ratio=1, clims=(min_mpc[it],max_mpc[it]), title = string(it))
    p1 = scatter!(p.x[p.phase.==0],p.y[p.phase.==0], color="red",   markersize=0.2, alpha=0.6, legend=false)
    p1 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="blue",  markersize=0.2, alpha=0.6, legend=false)
    p1 = scatter!(p.x[p.phase.==-1],p.y[p.phase.==-1], color="green",  markersize=0.2, alpha=0.6, legend=false)
    display(plot(p1))
    sleep(0.1)
end

end

@printf("\nGoing to run twice...\n")
@time Markers2D()
@time Markers2D()