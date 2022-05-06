# This script is benchmarked against Taras code - best advection
# Removed all @simd and @avx: this allows to defined main with @views
# Included Threads.@threads in front of each for statement
# No care taken about potential race conditions, see lines 206-207

# Initialisation
using Printf
using Statistics
using Plots
using LoopVectorization
using MAT
# pyplot(size = (750/1, 565/1))

path = @__DIR__

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
@views function Markers2D()
@printf("Running on %d thread(s)\n", Threads.nthreads())
# Import data
file = matopen( path * "/data41_benchmark.mat")
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
Threads.@threads for k=1:nmark
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
    Threads.@threads for j=1:ncy
        for i=1:ncx
            mpc[i,j] = 0.0
        end
    end
    Threads.@threads for k=1:nmark # @avx ne marche pas ici
        # Get the column:
        dstx = p.x[k] - xc[1];
        i    = Int64(round(ceil( (dstx/dx) + 0.5)));
        # Get the line:
        dsty = p.y[k] - yc[1];
        j    = Int64(round(ceil( (dsty/dy) + 0.5)));
        # Increment cell count
        mpc[i,j] += 1.0
    end

    # BASIC FUNCTION: Interpolate field from markers to centers
    weight = zeros(Float64,(ncx  ,ncy  )) 
    Threads.@threads for j=1:ncy
        for i=1:ncx
            phase[i,j] = 0
        end
    end 
    Threads.@threads for k=1:nmark # @avx ne marche pas ici
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
            # !!!!!!!
            # !!!!!!! WARNING: Here there is a risk of RACE condition
            phase[i,j]  += p.phase[k] * (1-dxm/dx)*(1-dym/dy);
            weight[i,j] +=              (1-dxm/dx)*(1-dym/dy);
            # !!!!!!!
        end
    end
    Threads.@threads for j=1:ncy
        for i=1:ncx
            phase[i,j] /= weight[i,j]
        end
    end 
    
    # BASIC FUNCTION: Marker advection with 4th order Roger-Gunther
    Threads.@threads for k=1:nmark
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
    # p1 = heatmap(xc, yc, transpose(phase),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
    p1 = heatmap(xc, yc, transpose(mpc),c=:inferno,aspect_ratio=1, clims=(min_mpc[it],max_mpc[it]), title = string(it))
    # p1 = quiver!(xc2[:], yc2[:], quiver=(VxC[:], VyC[:]))
    p1 = scatter!(p.x[p.phase.==0],p.y[p.phase.==0], color="red",   markersize=0.2, alpha=0.6, legend=false)
    p1 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="blue",  markersize=0.2, alpha=0.6, legend=false)
    p1 = scatter!(p.x[p.phase.==-1],p.y[p.phase.==-1], color="green",  markersize=0.2, alpha=0.6, legend=false)
    p2 = plot(1:it, min_mpc[1:it], color="blue", label="min.",foreground_color_legend = nothing, background_color_legend = nothing)
    p2 = plot!(1:it, max_mpc[1:it], color="red", label="max.")
    p2 = plot!(1:it, mean_mpc[1:it], color="green", label="mean.")
    p2 = plot!(1:it, tot_reseed[1:it], color="black", label="needed")
    display(plot(p1, p2))
    sleep(0.1)
end

end

@printf("\nGoing to run 3 times...\n")
@time Markers2D()
@time Markers2D()
@time Markers2D()