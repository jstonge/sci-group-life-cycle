"""
Steady states analysis
"""

using Pkg; Pkg.activate(".");
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots
using Plots.PlotMeasures
using OrderedCollections

include("../helpers.jl")
include("../sci-group-life-cycle.jl")

c(n, i; a=3) = n == i == 0 ? 0.95 : 0.95 * exp(-a*i / n)  # cost function
τ(n, i, α, β) = exp(-α + β*(1 - c(n, i))) # group benefits

function run_sci_group(p_lookup)
    u₀ = initialize_u0(N=20)
    prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, collect(values(p_lookup)))
    sol = solve(prob, Rosenbrock23(), saveat=1, reltol=1e-8, abstol=1e-8)
    return sol, frac_prog_in_groups(sol);
end

function plot_steady(params)
  param_lookup = OrderedDict(pname => p for (pname, p) in zip(param_lab, params))
  sol, out = run_sci_group(param_lookup)
  t_max, last_t = length(sol)-1, length(out[1,:])-1
  # param_string = join(["$(lab)=$(val)" for (lab, val) in zip(param_lab, params)], ';')
  plot(1:last_t, out[t_max,1:last_t], label=join(params, ", "), 
       legendtitle= join(param_lab, ",  "), legend=:outertop, marker=[:hex :d],
       palette = :Dark2_5)
end

function plot_steady!(params)
  param_lookup = OrderedDict(pname => p for (pname, p) in zip(param_lab, params))
  sol, out = run_sci_group(param_lookup)
  t_max, last_t = length(sol)-1, length(out[1,:])-1
  plot!(1:last_t, out[t_max,1:last_t], label=join(params, ", "), marker=[:hex :d])
end

tspan = (0., 4000) 
param_lab = ["μ", "νₙ", "νₚ", "α", "β", "a"]

params    = [0.5, 0.01, 0.01, 0.01, 0.1, 3]
plot_steady(params)

paramss = [
  [.05, .01 ,.5,.01 , .56, 3],
  [.5,  .01 ,.5,.01 , .56, 1],
  [.05, .01 ,.5,.01 , .76, 1],
  [.05, .01 ,.5,.01 , .76, 3]
]

for params in paramss
  plot_steady!(params)
end

xlabel!("group size")
ylabel!("proportion programmers")
plot!(size=(700, 450))
savefig("figs/prop_programmers_by_groupsize_steady.png")


# understanding what we're plotting

# v = [sol[t_max][1,5],sol[t_max][2,4],sol[t_max][3,3],sol[t_max][4,2],sol[t_max][5,1]]
# weights = [0/4, 1/4, 2/4, 3/4, 4/4]
# wsum(v, weights) / sum(v)

# scatter(1:length(nums), nums)
# plot!(1:length(nums), nums)


# ---------- how does group size occupying number change over time? ---------- #


# params = [.5, .01 ,.5, .01 , .56, 1]
# params = [.05, .01 ,.5,.01 , .56, 3]
# params = [0.5, 0.01, 0.01, 0.01, 0.1, 3]
# params = [.5,  .01 ,.5,.01 , .56, 1]
params = [.05, .01 ,.5,.01 , .76, 3]
sol, out = run_sci_group(params)
frac_prog_each_timestep = []
t_max, last_t = length(sol)-1, length(out[1,:])-1
for t=1:t_max
  occ_numbers = zeros(sum(size(first(sol))))
  nrows, ncols=size(sol[t])
  for i=1:nrows, j=1:ncols
    gsize = ((i-1)+(j-1))+1
    occ_numbers[gsize] += sol[t][i,j]
  end
  push!(frac_prog_each_timestep, occ_numbers)
end

y = [frac_prog_each_timestep[t][1] for t=1:t_max]
ps = plot(1:t_max, y, log, xaxis=:log, label="0", legendtitle= "gsize", legend=:outertopright)
for gsize in 2:3:21
  y = [frac_prog_each_timestep[t][gsize] for t=1:t_max]
  plot!(1:t_max, y, log, xaxis=:log, label="$(gsize-1)")
end
ps

title!("$(join(param_lab, "  ,  "))\n$(join(params, ","))")
ylabel!("Frac group size")
xlabel!("Time")

p2 = plot_steady(params)
xlabel!("group size")
ylabel!("proportion programmers")


plot(p2, ps, layout=(1, 2))
plot!(size=(1000, 450))
plot!(bottom_margin = 10mm)
plot!(left_margin = 10mm)
savefig("figs/$(join(params, "_")).png")

# ------------------- fraction of programmers by group size ------------------ #



# Plot showing `prop_prog ~ gsize (hue = timesize)`

function plot_sol(sol, p; outdir=nothing)  
  out = frac_prog_in_groups(sol)
  t_max = length(sol)-1
  last_t=length(out[1,:])-1

  param_lab = ["μ", "νₙ", "νₚ", "α", "β", "a", "p"]
  param_str = join(["$(pname)=$(p);" for (pname, p) in zip(param_lab, p)], ' ')
  ps=plot(1:last_t, out[2,1:last_t], label="t=2", legend=:outerright, top_margin = 20mm)
  for t=collect(5:5:30)
    plot!(1:last_t, out[t,1:last_t], label="t=$(t)") 
  end
  
  plot!(1:last_t, out[t_max,1:last_t], label="t=$(t_max)")
  
  title!("Many programmers in large groups while\nwe have few programmers in small groups\n($(param_str))")
  xlabel!("group size")
  ylabel!("proportion programmers")
  ylims!(0,1)
  plot!(size=(650,400))

  if isnothing(outdir) 
    println("Plotting")
    return ps
  else
    ps
    println("Writing to disk")
    savefig("$(outdir)/$(join(p, "_")).pdf")
  end
end

# fraction of programmer by group size
plot_sol(sol, params)  
xlims!(0,20)
plot!(size=(800,500))



frac_prog_each_timestep = []
nrows, ncols=size(first(sol))
for t=1:t_max
  # t=1
  nums = denums = zeros(nrows+ncols)
  
  for i=1:nrows, j=1:ncols
    p, n = i-1, j-1
    gsize = p+n
    
    if p+n > 0 
      nums[gsize] += (p / (gsize)) * sol[t][i,j]
      denums[gsize] += sol[t][i,j]
      
      if gsize == 10
        println((p, n, gsize, (p / (gsize)) * sol[t][i,j], sol[t][i,j]))
      end

    end
  end
  no_nans_out = map(x -> isnan(x) ? 0. : float(x), nums ./ denums)
  push!(frac_prog_each_timestep, no_nans_out)
end

frac_prog_each_timestep[2]

y = [frac_prog_each_timestep[t][1] for t=1:t_max]
ps = plot(1:t_max, y, log, xaxis=:log, label="0", legendtitle= "gsize", legend=:outertopright)
for gsize in 2:3:21
  y = [frac_prog_each_timestep[t][gsize] for t=1:t_max]
  plot!(1:t_max, y, log, xaxis=:log, label="$(gsize-1)")
end
ps
ylabel!("Frac group size")
xlabel!("Time")
plot!(size=(800, 450))
