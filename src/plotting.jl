using Pkg; Pkg.activate(".");
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots
using Plots.PlotMeasures
using OrderedCollections

include("helpers.jl")
include("sci-group-life-cycle.jl")

c(n, i; a=3) = n == i == 0 ? 0.95 : 0.95 * exp(-a*i / n)  # cost function
τ(n, i, α, β) = exp(-α + β*(1 - c(n, i))) # group benefits

function run_sci_group(p_lookup)
    u₀ = initialize_u0(N=20)
    prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, collect(values(p_lookup)))
    sol = solve(prob, Rosenbrock23(), saveat=1, reltol=1e-8, abstol=1e-8)
    return sol, frac_prog_in_groups(sol);
end


# --------------- fitness ~ transmission rate (by institutions) -------------- #

#!TODO: Similar analysis than for fitness ~ transmission rate (by institutions)
# function calc_group_benefits(sol; gsize)
#   # gsize=10
#   N, P = size(first(sol)) 
#   t_max=length(sol)-1
#   # benefits_over_time = zeros(t_max)
#   min_p = gsize <= (N-1) ? 1 : (gsize+1) - (N-1) 
#   for t=t_max
#     # t = t_max
#     out = []
#     for p=min_p:minimum([21, gsize+1])
#       # p=5
#       nb_prog = p-1
#       n = gsize-p+2
#       nb_non_prog = n - 1
#       push!(out, ( τ(nb_non_prog, nb_prog, α, β)*sol[t][n,p]) / sol[t][n,p] )
#     end
#   end
#   return out
# end


# ---------------- choosing the right group and cost functions --------------- #

c(n, i; a=1) = n == i == 0 ? 0.95 : 0.95 * exp(-a*i / n)  # cost function
τ(n, i, α, β) = exp(-α + β*(1 - c(n, i))) # group benefits
α, β =0.01, 0.1
# collect(.1:.33:1.)
a=1
ps = []
for α=.01:.033:.1, β=.1:.33:1.
  # p1=plot(x -> c(1,x, a=a), 0, 10, label="# non-prog=1")
  # plot!(x -> c(3,x, a=a), 0, 10, label="# non-prog=3")
  # plot!(x -> c(5,x, a=a), 0, 10, label="# non-prog=5")
  # plot!(x -> c(10,x, a=a), 0, 10, label="# non-prog=10")
  # xlabel!("# programmers")
  # ylabel!("c(n,p)")
  # title!("Cost function (a=$(a))")

  p2=plot(x -> exp(-α + β*(1-c(1,x, a=a))), 0, 20, label="# non-prog=1")
  xlabel!("# programmers")
  ylabel!("τ(n,p)")
  title!("e^(-$(α) + $(β)*(1-c))) & a=$(a)")
  plot!(x -> exp(-α + β*(1-c(3,x, a=a))), 0, 20, label="# non-prog=3")
  plot!(x -> exp(-α + β*(1-c(5,x, a=a))), 0, 20, label="# non-prog=5")
  plot!(x -> exp(-α + β*(1-c(10,x, a=a))), 0, 20, label="# non-prog=10")

  # push!(ps, plot(p1,p2,layout=(2,1)))
  push!(ps, p2)
end

plot(ps[1], ps[2], ps[3], 
     ps[4], ps[5], ps[6], 
     ps[7], ps[8], ps[9], 
     layout=(3,3), size=(1200,1200))
# ylims!(0.9, 2.5)
# ylims!(0.4, 1.)
# xlims!(0, 20)

# Note: 
# - No added benefits to go beyond 5 programmers, regardless of the number of non-programmers
# - As you add non-programmers, the group benefits of having more programmers decrease (prog are diluted).
#   Mostly because the cost is still high (n=10 => 0.2 vs n=3 => 0.).
# - When decreases a, cost stays longer nonzero, e.g. a=1, n=3, p=5 => c=.2 vs a=3, n=3, p=5 => c=0
# - When increasing non-coders benefits α, the curves do not change but you increase group benefits 
#   (α=.005,n=3,p=5 => τ≈exp(1.076) vs α=.1,n=3,p=5 => τ ≈ exp(.98))
# - Playing with β much more effective to have large group benefits than α