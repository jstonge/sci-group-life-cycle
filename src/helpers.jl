using CSV
using OrdinaryDiffEq:ODESolution

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table! s begin
      "--mu"
      arg_type = Float64
      default = 0.1
      help = "inflow new students-non coders"
      "--nu_n"
      arg_type = Float64
      default = 0.01
      help = "death rate non-coders"
      "--nu_p"
      arg_type = Float64
      default = 0.01
      help = "death rate coders"
      "--alpha"
      arg_type = Float64
      default = 0.01
      help = "benefits non coders"
      "--beta"
      arg_type = Float64
      default = 0.02
      help = "benefits coders"
      "--a"
      arg_type = Float64
      default = 3.
      help = "Parameter cost function, which is c(n, i; a=3) = n == i == 0 ? 0.95 : 0.95 * exp(-a*i / n)"
      "-o"
      help = "Output file for results"
    end

  return parse_args(s)
end


processing_sol1(x, n) = sum((collect(0:(n-1)) / n) .* x) / sum(x) 

function parse_sol(s::ODESolution)
    t1 = 1
    @assert s.u[t1] isa ArrayPartition
    N2 = length(first(s.u[t1].x)) # N2 can be anything; institution level, # predators, # programmers
    tmax = length(s)-1
    n2_indices = Dict()
    n2_indices_prop = Dict()
      
    for n₂=1:N2
      values = []
      values_prop = []
      for t=1:tmax
        n2_probs = s.u[t].x[n₂] # probs to find system in states n₂, regardless of n₁
        out = processing_sol1(n2_probs,N2)
        push!(values, out)
        out = sum(n2_probs)
        push!(values_prop, out)
      end
      n2_indices[n₂] = values
      n2_indices_prop[n₂] = values_prop
    end
    return n2_indices, n2_indices_prop
  end

  function plot_cost(c; a=3, p=5)
    p1=plot(x -> c(x, p, a=a), 1, 21, label="p=$(p)")
    xlabel!("# non programmers")
    
    p2=plot(x -> c(1, x, a=a), 0, 21, label="n=1")
    plot!(x -> c(5, x, a=a),  0, 21, label="n=5")
    plot!(x -> c(10, x, a=a), 0, 21, label="n=10")
    plot!(x -> c(15, x, a=a), 0, 21, label="n=15")
    xlabel!("# programmers")
    
    p3=plot(p1,p2, layout=(2,1))
    ylabel!("cost")
    title!("a=$(a)")  
    return p3
  end

function plot_tryptic_cost(c, p)
    p1=plot_cost(c, a=5, p=p)
    p2=plot_cost(c, a=3, p=p)
    p3=plot_cost(c, a=1, p=p)
    plot(p1,p2,p3, layout=(1,3), bottom_margin = 10mm)
    plot!(size=(700,500))
  end
  