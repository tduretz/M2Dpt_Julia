# This one can be used for benchmarking
#julia (--project) -O3 --check-bounds=no my_code.jl
# Initialisation
using Printf
using Statistics
using Plots
using LoopVectorization
using MAT
# pyplot(size = (750/1, 565/1))
Vizu     = 0
WriteOut = 1
style    = 4 # 1: Bilinear interp, 3: 1/3 - 2/3 trick, 4: new
noise    = 1
path = "/Users/imac/REPO_GIT/M2Dpt_Julia/Markers2D/AdvectionTaras/"
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
@views function VxVyFromPrNodes(Vxp ,Vyp, k, p, xce, yce, dx, dy, ncx, ncy)
    # Interpolate vy
    i = Int64((trunc( (p.x[k] - xce[1])/dx ) + 1.0));
    j = Int64((trunc( (p.y[k] - yce[1])/dy ) + 1.0));
    if i<1
        i=1
    elseif i>ncx+1
        i=ncx+1
    end
    if j<1
        j=1
    elseif j>ncy+1
        j = ncy+1
    end
    # Compute distances
    dxmj = p.x[k] - xce[i]
    dymi = p.y[k] - yce[j]
    # Compute weights
    wtmij   = (1.0-dxmj/dx)*(1.0-dymi/dy);
    wtmi1j  = (1.0-dxmj/dx)*(    dymi/dy);    
    wtmij1  = (    dxmj/dx)*(1.0-dymi/dy);
    wtmi1j1 = (    dxmj/dx)*(    dymi/dy);
    # Compute vx, vy velocity
    vxm = Vxp[i,j]*wtmij + Vxp[i,j+1]*wtmi1j + Vxp[i+1,j]*wtmij1 + Vxp[i+1,j+1]*wtmi1j1
    vym = Vyp[i,j]*wtmij + Vyp[i,j+1]*wtmi1j + Vyp[i+1,j]*wtmij1 + Vyp[i+1,j+1]*wtmi1j1
    return vxm, vym, i, j
end
########################################################################
function Markers2D()
gr(size = (750/1, 565/1))
@printf("Running on %d thread(s)\n", Threads.nthreads())
# Import data
file = matopen( path * "data41.mat")
Vx  = read(file, "Vx") # ACHTUNG THIS CONTAINS GHOST NODES AT NORTH/SOUTH
Vy  = read(file, "Vy") # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
Pt  = read(file, "Pt")
# dt  = read(file, "dt") # ACHTUNG THIS CONTAINS GHOST NODES AT EAST/WEST
# nt  = Int64(read(file, "ntimesteps"))
# xmi = read(file, "xm0")
# ymi = read(file, "ym0")
# if style==1
#     xm_Taras = read(file, "xm1")
#     ym_Taras = read(file, "ym1")
# elseif style==3
#     xm_Taras = read(file, "xm3")
#     ym_Taras = read(file, "ym3")
# elseif style==1
#     xm_Taras = read(file, "xm4")
#     ym_Taras = read(file, "ym4")
#     @printf("New scheme\n")
# end

Vxpt = read(file, "Vxp")
Vypt = read(file, "Vyp")

close(file)
########
itpw  = 1.0/3.0
Vizu  = 1
C     = 0.25
nt    = 20000
nout  = 1000
xmin  = 0
xmax  = 100
ymin  = 0
ymax  = 100
ncx   = 40
ncy   = 40
nmx   = 4                # 2 marker per cell in x
nmy   = 4                # 2 marker per cell in y
nmark = ncx*ncy*nmx*nmy; # total initial number of marker in grid
# ACTHUNG: For 1/3 - 2/3 interp one neede velocity on pressure points, including ghosts all around
Vxp = zeros(Float64,(ncx+2,ncy+2))
Vyp = zeros(Float64,(ncx+2,ncy+2))
Vxp[2:end-1,2:end-1] .= 0.5*(Vx[1:end-1,2:end-1] .+ Vx[2:end,2:end-1]) # for some reason Taras avoids BCs -  don't need this
Vyp[2:end-1,2:end-1] .= 0.5*(Vy[2:end-1,1:end-1] .+ Vy[2:end-1,2:end]) 

Vxp[2:end-1,1]   .= Vxp[2:end-1,2]
Vxp[2:end-1,end] .= Vxp[2:end-1,end-1] #
Vxp[1,:]           .= -Vxp[2,:]            # IMPORTANT: swelling enforces zero normal flux 
Vxp[end,:]         .= -Vxp[end-1,:]

Vyp[1,2:end-1]   .= Vyp[2,2:end-1]
Vyp[end,2:end-1] .= Vyp[end-1,2:end-1]
Vyp[:,1]           .= -Vyp[:,2]
Vyp[:,end]         .= -Vyp[:,end-1]

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
# Vx          = zeros(Float64,(ncx+1,ncy  ))
# Vy          = zeros(Float64,(ncx  ,ncy+1))
VxC         = zeros(Float64,(ncx  ,ncy  ))
VyC         = zeros(Float64,(ncx  ,ncy  ))
Vmag        = zeros(Float64,(ncx  ,ncy  )) 
mpc         = zeros(Float64,(ncx  ,ncy  )) # markers per cell
phase       = zeros(Float64,(ncx  ,ncy  ))
# 1D vectors for time series
min_mpc     = zeros(Float64,   nt )
max_mpc     = zeros(Float64,   nt )
mean_mpc    = zeros(Float64,   nt )
tot_reseed  = zeros(Float64,   nt )
nmark_add   = 0;
min_part_cell = 4
# set velocity
Threads.@threads for j=1:ncy
    for i=1:ncx
        VxC[i,j] = 0.5*(Vx[i,j+1] + Vx[i+1,j+1])
    end
end
Threads.@threads for j=1:ncy
    for i=1:ncx
        VyC[i,j] = 0.5*(Vy[i+1,j] + Vy[i+1,j+1])
    end
end
Threads.@threads for j=1:ncy
    for  i=1:ncx
        Vmag[i,j] = sqrt(VxC[i,j]^2 + VyC[i,j]^2)
    end
end
# Compute dt for Advection
dt = C * minimum((dx,dy)) / maximum( (maximum(Vx), maximum(Vy)) )
@printf( "dx = %2.2e --- dy = %2.2e --- dt = %2.2e\n", dx, dy, dt )
# Initialise markers
dxm, dym = dx/nmx, dy/nmy 
xm1d      =  LinRange(xmin+dxm/2, xmax-dxm/2, ncx*nmx)
ym1d      =  LinRange(ymin+dym/2, ymax-dym/2, ncy*nmy)
(xmi,ymi) = ([x for x=xm1d,y=ym1d], [y for x=xm1d,y=ym1d])
# Over allocate markers
# nmark_max = nmark;
# phm = zeros(Int64,   (nmark_max))
# xm  = zeros(Float64, (nmark_max)) 
# ym  = zeros(Float64, (nmark_max))
# Vxm = zeros(Float64, (nmark_max))
# Vym = zeros(Float64, (nmark_max))
#  for k=1:nmark
#     xm[k]  = xm2[k]
#     ym[k]  = ym2[k]
#     Vxm[k] =-ym2[k]
#     Vym[k] = xm2[k]
# end
# # xm[1:nmark]  = vec(xm2)
# # ym[1:nmark]  = vec(ym2)
# # Vxm[1:nmark] =-vec(ym2)
# # Vym[1:nmark] = vec(xm2)
# # xm  = xm2[:]
# # ym  = ym2[:]
# # Vxm =-ym2[:]
# # Vym = xm2[:]
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

if noise==1
    Threads.@threads for k=1:nmark
        # mshift = 
        p.x[k] += (rand()-0.5)*dxm
        p.y[k] += (rand()-0.5)*dym
    end
end

# RK4 weights
rkw = 1.0/6.0*[1.0 2.0 2.0 1.0] # for averaging
rkv = 1.0/2.0*[1.0 1.0 2.0 2.0] # for time stepping


# Time loop
for it=1:nt
    
    @printf("Time step #%04d\n", it)

    # Disable markers outside of the domain
    Threads.@threads for k=1:nmark
        if (p.x[k]<xmin || p.x[k]>xmax || p.y[k]<ymin || p.y[k]>ymax) 
            @inbounds p.phase[k] = -1
        end
    end

    # How many are outside? save indices for reuse
    nmark_out = 0
    Threads.@threads for k=1:nmark
        if p.phase[k] == -1
            @inbounds nmark_out += 1
        end
    end
    @printf("%04d markers out -- ideally %04d added markers\n", nmark_out, nmark_add)

#     # Save indices of particles to reuse
#     ind_reuse = zeros(Int64, nmark_out)
#     nmark_out = 0
#     @simd for k=1:nmark
#         if p.phase[k] == -1
#             @inbounds nmark_out            += 1
#             @inbounds ind_reuse[nmark_out]  = k
#         end
#     end

#     # find deficient cells
    
#     # add points with proper index

    # Count number of marker per cell
    Threads.@threads for j=1:ncy
        for i=1:ncx
            mpc[i,j] = 0.0
        end
    end
    for k=1:nmark # @avx ne marche pas ici
        if (p.phase[k]>=0)
            # Get the column:
            dstx = p.x[k] - xc[1];
            i    = Int64(round(ceil( (dstx/dx) + 0.5)));
            # Get the line:
            dsty = p.y[k] - yc[1];
            j    = Int64(round(ceil( (dsty/dy) + 0.5)));
            # Increment cell count
            mpc[i,j] += 1.0
        end
    end

    Threads.@threads for j=1:ncy
        for i=1:ncx
            if mpc[i,j] < min_part_cell
                nmark_add += min_part_cell # very weird need to use global!!!
            end
        end
    end

    # Interpolate phase on centers
    weight         = zeros(Float64,(ncx  ,ncy  )) 
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
            @inbounds phase[i,j]  += p.phase[k] * (1-dxm/dx)*(1-dym/dy);
            @inbounds weight[i,j] +=              (1-dxm/dx)*(1-dym/dy);
        end
    end
    Threads.@threads for j=1:ncy
        for i=1:ncx
            phase[i,j] /= weight[i,j]
        end
    end 
    
    # Marker advection with 4th order Roger-Gunther
    Threads.@threads for k=1:nmark
        if (p.phase[k]>=0)
            x0 = p.x[k];
            y0 = p.y[k];
            vx = 0.0
            vy = 0.0
            # Roger-Gunther loop
            for rk=1:4
                # Interp velocity from grid
                if style==1 # Bilinear velocity interp (original is Markers_divergence_ALLSCHEMES_RK4.m)
                    vxm = VxFromVxNodes(Vx, k, p, xv, yce, dx, dy, ncx, ncy, 0)
                    vym = VyFromVxNodes(Vy, k, p, xce, yv, dx, dy, ncx, ncy, 0)
                elseif style == 3
                    vxx = VxFromVxNodes(Vx, k, p, xv, yce, dx, dy, ncx, ncy, 0)
                    vyy = VyFromVxNodes(Vy, k, p, xce, yv, dx, dy, ncx, ncy, 0)
                    vxp, vyp, ix, iy = VxVyFromPrNodes(Vxp, Vyp, k, p, xce, yce, dx, dy, ncx, ncy)
                    vxm = itpw*vxp + (1.0-itpw)*vxx
                    vym = itpw*vyp + (1.0-itpw)*vyy
                elseif style == 4
                    vxm = VxFromVxNodes(Vx, k, p, xv, yce, dx, dy, ncx, ncy, 1)
                    vym = VyFromVxNodes(Vy, k, p, xce, yv, dx, dy, ncx, ncy, 1)
                end
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

    min_mpc[it]    = minimum(mpc)
    max_mpc[it]    = maximum(mpc)
    mean_mpc[it]   = mean(mpc)
    tot_reseed[it] = nmark_add

    if (Vizu == 1 && (it==1 || mod(it,nout)==0) )
        # p1 = heatmap(xc, yc, transpose(phase),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
        p1 = heatmap(xc, yc, transpose(mpc),c=:inferno, alpha=0.5, aspect_ratio=1, clims=(min_mpc[it],max_mpc[it]), title = string(it))
        # p1 = quiver!(xc2[:], yc2[:], quiver=(VxC[:], VyC[:]))
        p1 = scatter!(p.x[p.phase.==0],p.y[p.phase.==0], color="red",   markersize=0.2, alpha=0.6, legend=false)
        p1 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="blue",  markersize=0.2, alpha=0.6, legend=false)
        p1 = scatter!(p.x[p.phase.==-1],p.y[p.phase.==-1], color="green",  markersize=0.2, alpha=0.6, legend=false)
        p2 = plot(1:it, min_mpc[1:it], color="blue", label="min.",foreground_color_legend = nothing, background_color_legend = nothing)
        p2 = plot!(1:it, max_mpc[1:it], color="red", label="max.")
        p2 = plot!(1:it, mean_mpc[1:it], color="green", label="mean.")
        
        p3 = heatmap(xc, yc, transpose(mpc),c=:inferno, alpha=0.5, aspect_ratio=1, clims=(min_mpc[it],max_mpc[it]), title = string(it), xlims=(0,20), ylims=(80,100))
        p3 = scatter!(p.x[p.phase.==0],p.y[p.phase.==0], color="blue",  markersize=0.2, alpha=0.6, legend=false)
        p3 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="green",  markersize=0.2, alpha=0.6, legend=false)

        p4 = plot(1:it, tot_reseed[1:it], color="black", label="needed")
        display(plot(p1, p2, p3, p4))
        sleep(0.1)
    end

end

# @printf("Style %d - difference with Taras: %2.2e\n", style, mean(xm_Taras.-p.x[:]) )

it = nt

if (WriteOut == 1)
    file = matopen( path * "output_style" * string(style) * ".mat", "w")
    write(file, "xv", convert(Array{Float64,1},xv))
    write(file, "yv", convert(Array{Float64,1},yv))
    write(file, "xm", p.x)
    write(file, "ym", p.y)
    write(file, "phm", p.phase)
    write(file, "mpc", mpc)
    write(file, "min_mpc", min_mpc)
    write(file, "max_mpc", max_mpc)
    write(file, "mean_mpc", mean_mpc)
    write(file, "tot_reseed", tot_reseed)
    close(file)
end

end

@time Markers2D()
# @time Markers2D()