using Printf
using Plots
using Base.Threads
# plotlyjs() # this takes too long to precompile
gr()
#######################################################################
struct Markers
    x         ::  Array{Float64,1}
    y         ::  Array{Float64,1}
    phase     ::  Array{Float64,1}
end
########################################################################
@views function MainReduction2D()
    @printf("Running on %d thread(s)\n", nthreads())
    # Parameters
    xmin  = 0
    xmax  = 1
    ymin  = 0
    ymax  = 1
    ncx   = 40
    ncy   = 40
    nmx   = 4                # 2 marker per cell in x
    nmy   = 4                # 2 marker per cell in y
    nmark = ncx*ncy*nmx*nmy; # total initial number of marker in grid
    R     = 0.15
    L     = xmax-xmin;
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
    (xvx2,yvx2) = ([x for x=xv,y=yce], [y for x=xv,y=yce]);
    (xvy2,yvy2) = ([x for x=xce,y=yv], [y for x=xce,y=yv]);
    # Allocations
    phase       = zeros(Float64,(ncx  ,ncy  ))
    phase_th    = zeros(Float64,(nthreads(), ncx  ,ncy  )) # per thread
    weight      = zeros(Float64,(ncx  ,ncy  ))
    weight_th   = zeros(Float64,(nthreads(), ncx  ,ncy  )) # per thread
    # Initialise markers
    dxm, dym    = dx/nmx, dy/nmy 
    xm1d        =  LinRange(xmin+dxm/2, xmax-dxm/2, ncx*nmx)
    ym1d        =  LinRange(ymin+dym/2, ymax-dym/2, ncy*nmy)
    (xmi,ymi)   = ([x for x=xm1d,y=ym1d], [y for x=xm1d,y=ym1d])
    p           = Markers( vec(xmi), vec(ymi), zeros(Float64, nmark) )
    # Define phase on markers
    @threads for k=1:nmark
        if ((p.x[k]-L/2)^2 + (p.y[k]-3/4*L)^2 < R^2) 
            p.phase[k] = 1
        end
    end
    #
    #  C'est par ici que ca se passe
    #
    # Loop over particles - find local cell and apply bilinear interpolation
    @threads for k=1:nmark 
        # Get the column:
        dstx = p.x[k] - xc[1];
        i    = Int(ceil( (dstx/dx) + 0.5));
        # Get the line:
        dsty = p.y[k] - yc[1];
        j    = Int(ceil( (dsty/dy) + 0.5));
        # Relative distances
        dxm = 2.0*abs( xc[i] - p.x[k]);
        dym = 2.0*abs( yc[j] - p.y[k]);
        # Increment cell counts
        phase_th[threadid(),i,j]  += p.phase[k] * (1.0-dxm/dx)*(1.0-dym/dy);
        weight_th[threadid(),i,j] +=              (1.0-dxm/dx)*(1.0-dym/dy);
    end
    # Sum contributions from different threads
    @threads for j=1:ncy
        for i=1:ncx
            for ith=1:nthreads()
                if (ith==1)
                    phase[i,j] = 0.0
                    weight[i,j] = 0.0
                end
                phase[i,j]  += phase_th[ith,i,j]
                weight[i,j] += weight_th[ith,i,j]
            end
        end
    end
    # Final division (interpolation weights)
    @threads for j=1:ncy
        for i=1:ncx
            phase[i,j] = phase[i,j] / weight[i,j]
        end
    end

    # Visu
    p1 = heatmap(xc, yc, transpose(phase),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
    p2 = scatter(p.x[p.phase.==0],p.y[p.phase.==0], color="blue",  markersize=0.2, alpha=0.6, legend=false,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
    p2 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="red",  markersize=0.2, alpha=0.6, legend=false)
    display(plot(p1,p2))
    sleep(0.1)

end

@time MainReduction2D()