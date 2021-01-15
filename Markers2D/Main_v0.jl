# Initialisation
using Printf
using Statistics
using Plots
using LoopVectorization
gr(size = (750/4, 565/4))
########################################################################
Vizu  = 1
C     = 0.25
nt    = 10
xmin  = -.5
xmax  = 0.5
ymin  = -.5
ymax  = 0.5
ncx   = 10
ncy   = 20
nmx   = 2                # 2 marker per cell in x
nmy   = 2                # 2 marker per cell in y
nmark = ncx*ncy*nmx*nmy; # total initial number of marker in grid
# Spacing
dx, dy  = (xmax-xmin)/ncx, (ymax-ymin)/ncy
# 1D coordinates 
xc          = LinRange(xmin+dx/2, xmax-dx/2, ncx)
yc          = LinRange(ymin+dy/2, ymax-dy/2, ncy)
xv          = LinRange(xmin, xmax, ncx+1)
yv          = LinRange(ymin, ymax, ncy+1)
# 2D mesh
(xc2,yc2)   = ([x for x=xc,y=yc], [y for x=xc,y=yc]);
(xv2,yv2)   = ([x for x=xv,y=yv], [y for x=xv,y=yv]);
(xvx2,yvx2) = ([x for x=xv,y=yc], [y for x=xv,y=yc]);
(xvy2,yvy2) = ([x for x=xc,y=yv], [y for x=xc,y=yv]);
# 2D tables
Vx          = zeros(Float64,(ncx+1,ncy  ))
Vy          = zeros(Float64,(ncx  ,ncy+1))
VxC         = zeros(Float64,(ncx  ,ncy  ))
VyC         = zeros(Float64,(ncx  ,ncy  ))
Vmag        = zeros(Float64,(ncx  ,ncy  )) 
mpc         = zeros(Float64,(ncx  ,ncy  )) # markers per cell
phase       = zeros(Float64,(ncx  ,ncy  ))
# set velocity
@avx for j=1:ncy, i=1:ncx+1
     Vx[i,j] =-yvx2[i,j]
end
@avx for j=1:ncy+1, i=1:ncx
    Vy[i,j] = xvy2[i,j]
end
@avx for j=1:ncy, i=1:ncx
    VxC[i,j] = 0.5*(Vx[i,j] + Vx[i+1,j])
end
@avx for j=1:ncy, i=1:ncx
    VyC[i,j] = 0.5*(Vy[i,j] + Vy[i,j+1])
end
@avx for j=1:ncy, i=1:ncx
    Vmag[i,j] = sqrt(VxC[i,j]^2 + VyC[i,j]^2)
end
# Compute dt for Advection
dt = C * minimum((dx,dy)) / maximum( (maximum(Vx), maximum(Vy)) )
@printf( "dx = %2.2e --- dy = %2.2e --- dt = %2.2e\n", dx, dy, dt )
# Initialise markers
dxm, dym = dx/nmx, dy/nmy 
mutable struct Markers
    x         ::  Array{Float64,1}
    y         ::  Array{Float64,1}
    Vx        ::  Array{Float64,1}
    Vy        ::  Array{Float64,1}
    phase     ::  Array{Int64,1}
end
xm1d      =  LinRange(xmin+dxm/2, xmax-dxm/2, ncx*nmx)
ym1d      =  LinRange(ymin+dym/2, ymax-dym/2, ncy*nmy)
(xm2,ym2) = ([x for x=xm1d,y=ym1d], [y for x=xm1d,y=ym1d])
p         =  Markers( vec(xm2), vec(ym2), -vec(ym2), vec(xm2), zeros(Int64, nmark) )
# define phase
for k=1:nmark
    if (p.y[k]<0) 
        p.phase[k] = 1
    end
end

# Time loop
for it=1:nt
    @printf("Time step #%04d\n", it)

    # Disable markers outside of the domain
    for k=1:nmark
        if (p.x[k]<xmin || p.x[k]>xmax || p.y[k]<ymin || p.y[k]>ymax) 
            p.phase[k] = -1
        end
    end

    # How many are outside? save indices for reuse

    # find deficient cells
    
    # add points with proper index

    # Count number of marker per cell
    @avx for j=1:ncy, i=1:ncx
        mpc[i,j] = 0
    end
    for k=1:nmark # @avx ne marche pas ici
        if (p.phase[k]>=0)
            # Get the column:
            dstx = p.x[k] - xc[1];
            i    = Int(ceil( (dstx/dx) + 0.5));
            # Get the line:
            dsty = p.y[k] - yc[1];
            j    = Int(ceil( (dsty/dy) + 0.5));
            # Increment cell count
            mpc[i,j] += 1;
        end
    end

    # Interpolate phase on centers
    weight         = zeros(Float64,(ncx  ,ncy  )) 
    @avx for j=1:ncy, i=1:ncx
        phase[i,j] = 0
    end 
    for k=1:nmark # @avx ne marche pas ici
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
            phase[i,j]  += p.phase[k] * (1-dxm/dx)*(1-dym/dy);
            weight[i,j] +=              (1-dxm/dx)*(1-dym/dy);
        end
    end
    @avx for j=1:ncy, i=1:ncx
       phase[i,j] /= weight[i,j]
    end 
    
    # Update coordinate with Roger-Gunther 1
    @avx for k=1:nmark
        p.x[k] += dt*p.Vx[k]
        p.y[k] += dt*p.Vy[k]
    end

    if (Vizu == 1)
        p1 = heatmap(xc, yc, transpose(phase),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
        # p1 = heatmap(xc, yc, transpose(Vmag),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
        # p1 = quiver!(xc2[:], yc2[:], quiver=(VxC[:], VyC[:])) # PROBLEM: Does not Work at all
        # p1 = scatter!(p.x,p.y, shape=:o, alpha=0.5, markersize=1, legend=false)
        display(plot( p1 ))
        sleep(0.1)
    end
end