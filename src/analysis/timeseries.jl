"""
Plots over time
"""

using Pkg; Pkg.activate(".");
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots
using Plots.PlotMeasures
using OrderedCollections

include("../helpers.jl")
include("../sci-group-life-cycle.jl")

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
xlims!(0,40)
plot!(size=(800,500))
