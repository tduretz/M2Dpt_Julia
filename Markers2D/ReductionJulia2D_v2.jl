using Plots
import Base.Threads: @threads, @sync, @spawn, nthreads, threadid

struct Markers
    x::Vector{Float64}
    y::Vector{Float64}
    phase::Vector{Float64}
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

@views function MainReduction2D()

    println("Running on $(nthreads()) thread(s)")

    # Parameters
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    ncx = 40
    ncy = 40
    nmx = 4                # 2 marker per cell in x
    nmy = 4                # 2 marker per cell in y
    nmark = ncx * ncy * nmx * nmy # total initial number of marker in grid
    R = 0.15
    L = xmax - xmin
    # Spacing
    dx, dy = (xmax - xmin) / ncx, (ymax - ymin) / ncy
    # 1D coordinates 
    xc = LinRange(xmin + dx / 2, xmax - dx / 2, ncx)
    yc = LinRange(ymin + dy / 2, ymax - dy / 2, ncy)
    # Allocations
    phase     = zeros(Float64, ncx, ncy)
    phase_th  = [similar(phase) for _ = 1:nthreads()] # per thread
    weight    = zeros(Float64, (ncx, ncy))
    weight_th = [similar(weight) for _ = 1:nthreads()] # per thread
    # Initialise markers
    dxm, dym = dx / nmx, dy / nmy
    xm1d = LinRange(xmin + dxm / 2, xmax - dxm / 2, ncx * nmx)
    ym1d = LinRange(ymin + dym / 2, ymax - dym / 2, ncy * nmy)
    (xmi, ymi) = ([x for x in xm1d, y in ym1d], [y for x in xm1d, y in ym1d])
    p = Markers(vec(xmi), vec(ymi), zeros(Float64, nmark))
    # Define phase on markers
    @threads for k = 1:nmark
        if ((p.x[k] - L / 2)^2 + (p.y[k] - 3 / 4 * L)^2 < R^2)
            p.phase[k] = 1
        end
    end
    @threads for k=1:nmark
        p.x[k] += 1*(rand()-0.5)*dxm
        p.y[k] += 1*(rand()-0.5)*dym
    end
    # Call routine de Pierre
    @time  Markers2Cells(p,nmark,phase_th,phase,weight_th, weight,xc,yc,dx,dy,ncx,ncy)
   
    # liste de Pierre
    liste = hcat([[Int[] for i in 1:ncx] for j in 1:ncy]...)

    for k in 1:nmark
        dstx = p.x[k] - xc[1]
        i = ceil(Int, dstx / dx + 0.5)
        dsty = p.y[k] - yc[1]
        j = ceil(Int, dsty / dy + 0.5)
        push!(liste[i,j], k)
    end
    println(size(liste))
    for i in 1:ncx
         for j in 1:ncy
    # println(liste[i,j][:])
    println(size(liste[i,j]))
         end
        end
    # Visu
    p1 = heatmap(xc, yc, transpose(phase),c=:inferno,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
    p2 = scatter(p.x[p.phase.==0],p.y[p.phase.==0], color="blue",  markersize=0.2, alpha=0.6, legend=false,aspect_ratio=1, xlims=(xmin,xmax), ylims=(ymin,ymax))
    p2 = scatter!(p.x[p.phase.==1],p.y[p.phase.==1], color="red",  markersize=0.2, alpha=0.6, legend=false)
    return plot(p1, p2)
end

result = MainReduction2D()
# result = MainReduction2D()
# result = MainReduction2D()
# result = MainReduction2D()
# result = MainReduction2D()
# result = MainReduction2D()
# result = MainReduction2D()

display(result)
