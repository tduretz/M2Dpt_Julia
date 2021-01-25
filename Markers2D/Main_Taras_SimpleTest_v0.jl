# This one can be used for benchmarking
#julia (--project) -O3 --check-bounds=no my_code.jl
# Initialisation
using Printf
using Statistics
using Plots
using LoopVectorization
using MAT
using LinearAlgebra
using Base.Threads
# gr()
pyplot(size = (750/1, 565/1))
# plotlyjs()
Vizu     = 1
WriteOut = 1
style    = 4 # 1: Bilinear interp, 3: 1/3 - 2/3 trick, 4: new, 5: divergence-free RBF
noise    = 0
path = "/Users/imac/REPO_GIT/M2Dpt_Julia/Markers2D/AdvectionTaras/"
########################################################################
struct Markers
    x         ::  Array{Float64,1}
    y         ::  Array{Float64,1}
    phase     ::  Array{Float64,1}
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
    return vxm, vym
end
@views function PrecomputeInterpolant( xce, yce, epsi )
    i = 1
    j = 1
    x_locp = [xce[i]; xce[i+1]; xce[i  ]; xce[i+1]];
    y_locp = [yce[j]; yce[j  ]; yce[j+1]; yce[j+1]];

    x_ij   = zeros(4,4);
    y_ij   = zeros(4,4);
    r_ij   = zeros(4,4);

    for j=1:4
        for i=1:4
            xx  = abs(x_locp[i] - x_locp[j]);
            yy  = abs(y_locp[i] - y_locp[j]);
            x_ij[i,j] = xx;
            y_ij[i,j] = yy;
            r_ij[i,j] = sqrt(xx^2 + yy^2);
        end
    end

    rp2 = zeros(4,4);
    phi11 = zeros(4,4);
    phi12 = zeros(4,4);
    phi21 = zeros(4,4);
    phi22 = zeros(4,4);

    @. rp2   = r_ij.^2;
    @. phi11 = -2*epsi*(2*epsi*y_ij.^2 - 1) .* exp(-epsi.*rp2);
    @. phi12 =  4*epsi^2*x_ij.*y_ij.*exp(-epsi.*rp2);
    @. phi21 = phi12;
    @. phi22 = -2*epsi*(2*epsi*x_ij.^2 - 1) .* exp(-epsi.*rp2);

    A      = [(phi11) (phi12); (phi21) (phi22);];
    Afact = factorize(A)
    return Afact
end

@views function VxVyFromPrNodesDivFreeRBF(Vxp ,Vyp, k, p, xce, yce, dx, dy, ncx, ncy, Afact, epsi)
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

    vxm = 0.0
    vym = 0.0

    vxp    = [Vxp[i,j]; Vxp[i+1,j]; Vxp[i,j+1]; Vxp[i+1,j+1]];
    vyp    = [Vyp[i,j]; Vyp[i+1,j]; Vyp[i,j+1]; Vyp[i+1,j+1]];
    x_locp = [xce[i]; xce[i+1]; xce[i  ]; xce[i+1]];
    y_locp = [yce[j]; yce[j  ]; yce[j+1]; yce[j+1]];

    d      = [vxp; vyp];# + [0.5*ones(size(Vxp)); 0.5*ones(size(Vyp))];
    c      = Afact\d;
    cx     = c[1:4];
    cy     = c[5:end];

    x      = zeros(4,1);
    y      = zeros(4,1);
    rp1    = zeros(4,1);
    phi11  = zeros(4,1);
    phi12  = zeros(4,1);
    phi21  = zeros(4,1);
    phi22  = zeros(4,1);
    @. x      = abs(x_locp - p.x[k])
    @. y      = abs(y_locp - p.y[k])
    @. rp1    = x.^2.0 + y.^2.0;
    @. phi11 = -2.0*epsi*(2.0*epsi*y.^2.0 - 1.0) .* exp(-epsi.*rp1);
    @. phi12 =  4.0*epsi^2.0*x.*y.*exp(-epsi*rp1);
    @. phi21 = phi12;
    @. phi22 = -2.0*epsi*(2.0*epsi*x.^2.0 - 1.0) .* exp(-epsi.*rp1);

    dphi11dx  = zeros(4,1);
    dphi21dx  = zeros(4,1);
    dphi12dy  = zeros(4,1);
    dphi22dy  = zeros(4,1);
    @. dphi11dx = 4.0*epsi^2.0*x*(2.0*y^2.0*epsi - 1.0)*exp(-epsi*rp1);
    @. dphi21dx = -8.0*x^2.0*y*epsi^3.0*exp(-epsi*rp1) + 4.0*y*epsi^2.0*exp(-epsi*rp1);
    @. dphi12dy = -8.0*x*y^2*epsi^3.0*exp(-epsi*rp1) + 4.0*x*epsi^2.0*exp(-epsi*rp1);
    @. dphi22dy = 4.0*y*epsi^2.0*(2.0*x^2.0*epsi - 1.0)*exp(-epsi*rp1);

    vxm    = sum(phi11.*cx) + sum(phi21.*cy);
    vym    = sum(phi12.*cx) + sum(phi22.*cy);
    dvxdx  = sum(dphi11dx.*cx) + sum(dphi21dx.*cy);
    dvydy  = sum(dphi12dy.*cx) + sum(dphi22dy.*cy);
    div    = dvxdx + dvydy;

    # if abs(div)>1e-20
        # if threadid()==1 && p.phase[k] == 1
        #     @printf("vx = %2.2e --- vy = %2.2e --- div =%2.2e\n", vxm, vym, div)
        # end
    # end

    return vxm, vym
end
@views function Markers2Cells(p,nmark,phase_th,phase,weight_th, weight,xc,yc,dx,dy,ncx,ncy)
    chunks = Iterators.partition(1:nmark, nmark ÷ nthreads())
    @sync for chunk in chunks
        @spawn begin
            tid = threadid()
            fill!(phase_th[tid], 0)
            fill!(weight_th[tid], 0)
            for k in chunk
                # Get the column:
                dstx = p.x[k] - xc[1]
                i = ceil(Int, dstx / dx + 0.5)
                # Get the line:
                dsty = p.y[k] - yc[1]
                j = ceil(Int, dsty / dy + 0.5)
                # Relative distances
                dxm = 2.0 * abs(xc[i] - p.x[k])
                dym = 2.0 * abs(yc[j] - p.y[k])
                # Increment cell counts
                area = (1.0 - dxm / dx) * (1.0 - dym / dy)
                phase_th[tid][i, j] += p.phase[k] * area
                weight_th[tid][i, j] += area
            end
        end
    end
    phase  .= reduce(+, phase_th)
    weight .= reduce(+, weight_th)
    phase ./= weight
    return
end
@views function CountMarkersPerCell(p,nmark,mpc,mpc_th,xc,yc,dx,dy)
chunks = Iterators.partition(1:nmark, nmark ÷ nthreads())
    @sync for chunk in chunks
        @spawn begin
            tid = threadid()
            fill!(mpc_th[tid], 0)
            for k in chunk
                # Get the column:
                dstx = p.x[k] - xc[1]
                i    = ceil(Int, dstx / dx + 0.5)
                # Get the line:
                dsty = p.y[k] - yc[1]
                j    = ceil(Int, dsty / dy + 0.5)
                # Increment cell counts
                mpc_th[tid][i, j] += 1
            end
        end
    end
    mpc .= reduce(+, mpc_th)
    return
end
@views function SetUpVelocity(Vx,VxC,Vxv,Vy,VyC,Vyv,xvx2,yvx2,xvy2,yvy2, ncx,ncy,L,sign)
@. Vx = sign*(cos(pi*(xvx2 - L/2)) * sin(pi*(yvx2 - L/2)))
@. Vy = sign*(-sin(pi*(xvy2 - L/2)) * cos(pi*(yvy2 - L/2)))
@threads for j=1:ncy
    for i=1:ncx
        VxC[i,j] = 0.5*(Vx[i,j+1] + Vx[i+1,j+1])
    end
end
@threads for j=1:ncy
    for i=1:ncx
        VyC[i,j] = 0.5*(Vy[i+1,j] + Vy[i+1,j+1])
    end
end
@. Vxv = 0.5*(Vx[:,2:end]+Vx[:,1:end-1])
@. Vyv = 0.5*(Vy[2:end,:]+Vy[1:end-1,:])
end
########################################################################
function Markers2D()
@printf("Running on %d thread(s)\n", nthreads())

anim = Animation();
########
itpw  = 1.0/3.0
Vizu  = 1
C     = 0.25
nt    = 5000 #20000
time  = 0.0
nout  = 10
xmin  = 0
xmax  = 1
ymin  = 0
ymax  = 1
ncx   = 40
ncy   = 40
nmx   = 4                # 2 marker per cell in x
nmy   = 4                # 2 marker per cell in y
nmark = ncx*ncy*nmx*nmy; # total initial number of marker in grid
epsi  = (1.0/8.0)^2
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
# 2D tables
Vx          = zeros(Float64,(ncx+1,ncy+2))
Vy          = zeros(Float64,(ncx+2,ncy+1))
VxC         = zeros(Float64,(ncx  ,ncy  ))
VyC         = zeros(Float64,(ncx  ,ncy  ))
Vxv         = zeros(Float64,(ncx+1,ncy+1))
Vyv         = zeros(Float64,(ncx+1,ncy+1))
Vmag        = zeros(Float64,(ncx  ,ncy  )) 
# Allocations
phase       = zeros(Float64, ncx, ncy)
phase_th    = [similar(phase) for _ = 1:nthreads()] # per thread
weight      = zeros(Float64, (ncx, ncy))
weight_th   = [similar(weight) for _ = 1:nthreads()] # per thread
mpc         = zeros(Float64,(ncx  ,ncy  ))           # markers per cell
mpc_th      = [similar(mpc) for _ = 1:nthreads()]    # per thread
# 1D vectors for time series
min_mpc     = zeros(Float64,   nt )
max_mpc     = zeros(Float64,   nt )
mean_mpc    = zeros(Float64,   nt )
tot_reseed  = zeros(Float64,   nt )
nmark_add   = 0;
min_part_cell = 4
Afact = PrecomputeInterpolant( xce, yce, epsi )
# set velocity
L    = xmax-xmin
R    = 0.15
sign = 1.0;
SetUpVelocity(Vx,VxC,Vxv,Vy,VyC,Vyv,xvx2,yvx2,xvy2,yvy2, ncx,ncy,L,sign)
# Compute dt for Advection
dt = C * minimum((dx,dy)) / maximum( (maximum(Vx), maximum(Vy)) )
@printf( "dx = %2.2e --- dy = %2.2e --- dt = %2.2e\n", dx, dy, dt )
# Initialise markers
dxm, dym = dx/nmx, dy/nmy 
xm1d      =  LinRange(xmin+dxm/2, xmax-dxm/2, ncx*nmx)
ym1d      =  LinRange(ymin+dym/2, ymax-dym/2, ncy*nmy)
(xmi,ymi) = ([x for x=xm1d,y=ym1d], [y for x=xm1d,y=ym1d])
xm   = vec(xmi)
ym   = vec(ymi)
phm  = zeros(Int64,   size(xm))
p    = Markers( xm, ym, phm )
# define phase
@threads for k=1:nmark
    if ((p.x[k]-L/2)^2 + (p.y[k]-3/4*L)^2 < R^2) 
        p.phase[k] = 1
    end
end

if noise==1
    @threads for k=1:nmark
        # mshift = 
        p.x[k] += (rand()-0.5)*dxm
        p.y[k] += (rand()-0.5)*dym
    end
end

# RK4 weights
rkw = 1.0/6.0*[1.0 2.0 2.0 1.0] # for averaging
rkv = 1.0/2.0*[1.0 1.0 2.0 2.0] # for time stepping

# Time loop
for it=0:nt
    
    @printf("Time step #%04d\n", it)
    if it==nt/2
        sign = -1.0;
        SetUpVelocity(Vx,VxC,Vxv,Vy,VyC,Vyv,xvx2,yvx2,xvy2,yvy2, ncx,ncy,L,sign)
    end

    # Disable markers outside of the domain
    @threads for k=1:nmark
        if (p.x[k]<xmin || p.x[k]>xmax || p.y[k]<ymin || p.y[k]>ymax) 
            @inbounds p.phase[k] = -1
        end
    end

    # How many are outside? save indices for reuse
    nmark_out_th = zeros(Int64, nthreads())
    @threads for k=1:nmark
        if p.phase[k] == -1
            nmark_out_th[threadid()] += 1
        end
    end
    nmark_out = 0
    for ith=1:nthreads()
        nmark_out += nmark_out_th[ith]
    end
    @printf("%d markers out\n", nmark_out)

    ###########################################################################

    CountMarkersPerCell(p,nmark,mpc,mpc_th,xc,yc,dx,dy)

    ###########################################################################

    Markers2Cells(p,nmark,phase_th,phase,weight_th, weight,xc,yc,dx,dy,ncx,ncy)

    ###########################################################################

    # Marker advection with 4th order Roger-Gunther
    @threads for k=1:nmark
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
                    vxp, vyp = VxVyFromPrNodes(Vxp, Vyp, k, p, xce, yce, dx, dy, ncx, ncy)
                    vxm = itpw*vxp + (1.0-itpw)*vxx
                    vym = itpw*vyp + (1.0-itpw)*vyy
                elseif style == 4
                    vxm = VxFromVxNodes(Vx, k, p, xv, yce, dx, dy, ncx, ncy, 1)
                    vym = VyFromVxNodes(Vy, k, p, xce, yv, dx, dy, ncx, ncy, 1)
                elseif style == 5   
                    # vxm, vym = VxVyFromPrNodesDivFreeRBF(Vxp, Vyp, k, p, xce, yce, dx, dy, ncx, ncy)
                    vxm, vym = VxVyFromPrNodesDivFreeRBF(Vxv, Vyv, k, p, xv, yv, dx, dy, ncx-1, ncy-1, Afact, epsi)
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

    # min_mpc[it]    = minimum(mpc)
    # max_mpc[it]    = maximum(mpc)
    # mean_mpc[it]   = mean(mpc)
    # tot_reseed[it] = nmark_add

    if (Vizu == 1 && (it==1 || mod(it,nout)==0) )
        p1 = heatmap(xc, yc, transpose(phase),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax),title = string(time))
        # p1 = heatmap(xc, yc, transpose(mpc),c=:inferno, alpha=0.5, aspect_ratio=1, clims=(min_mpc[it],max_mpc[it]), title = string(it))
        # p1 = quiver!(xc2[:], yc2[:], quiver=(VxC[:], VyC[:]))
        # p1 = scatter(p.x[p.phase.==0],p.y[p.phase.==0], color="red",   markersize=0.2, alpha=0.6, legend=false)
        # p1 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="blue",  markersize=0.2, alpha=0.6, legend=false)
        # p1 = scatter!(p.x[p.phase.==-1],p.y[p.phase.==-1], color="green",  markersize=0.2, alpha=0.6, legend=false)
        # p2 = plot(1:it, min_mpc[1:it], color="blue", label="min.",foreground_color_legend = nothing, background_color_legend = nothing,title = string(it))
        # p2 = plot!(1:it, max_mpc[1:it], color="red", label="max.")
        # p2 = plot!(1:it, mean_mpc[1:it], color="green", label="mean.")
        
        # p3 = heatmap(xc, yc, transpose(mpc),c=:inferno, alpha=0.5, aspect_ratio=1, clims=(min_mpc[it],max_mpc[it]), title = string(time))
        # p3 = scatter!(p.x[p.phase.==0],p.y[p.phase.==0], color="blue",  markersize=0.2, alpha=0.6, legend=false)
        # p3 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="green",  markersize=0.2, alpha=0.6, legend=false)

        # p4 = plot(1:it, tot_reseed[1:it], color="black", label="needed")
        # display(plot(p1, p2))
        # # display(plot(p1))   
        # sleep(0.1)
        frame(anim)
    end

    time          += sign*dt

end

gif(anim, "SimpleAdvectionTest.gif", fps = 15)

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