using Pkg; Pkg.activate(".");
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots
using Plots.PlotMeasures

include("helpers.jl")


function initialize_u0(;N=20, M::Int=100, p::Float64=0.01)
  N_plus_one = N+1
  G = zeros(N_plus_one, N_plus_one)
  for _ in 1:M
    ℓ = rand(1:N_plus_one)
    i = sum(rand(Binomial(1, p), N_plus_one))
    G[ℓ, i+1] += 1
  end
  G = G ./ M
  return G
end

c(n, i; a=3) = n == i == 0 ? 0.95 : 0.95 * exp(-a*i / n)  # cost function
τ(n, i, α, β) = exp(-α + β*(1 - c(n, i))) # group benefits

function life_cycle_research_groups!(du, u, p, t)
  N, P = size(u) # Note that there can be no coders but not non-coders
  G = u # G for groups
  μ, νₙ, νₚ, α, β, a = p
  for n=1:N, i=1:P
    coder, noncoder = i-1, n-1 
    du[n,i] = 0

    noncoder > 0 && ( du[n,i] += μ*G[n-1,i] ) # non-prog repro input
    n < N && ( du[n,i] -= G[n,i]*μ )                       # non-prog repro output

    n < N && ( du[n,i] += G[n+1,i]*νₙ*(noncoder+1) ) # non-prog death input
    du[n,i] -= G[n,i]*noncoder*νₙ                    # non-prog death output
    
    i < P && ( du[n,i] += G[n,i+1]*νₚ*(coder+1) ) # prog death input
    du[n,i] -= G[n,i]*coder*νₚ                    # prog death output
    
    (coder > 0 && n < N) && ( du[n,i] += G[n+1,i-1]*(noncoder+1)*τ(noncoder+1, coder-1, α, β)*(1-c(noncoder+1,coder-1, a=a)) ) # non-prog to prog predation input
    i < P && ( du[n,i] -= G[n,i]*noncoder*τ(noncoder, coder, α, β)*(1-c(noncoder,coder, a=a)) ) # non-prog to prog predation output
    
    i < P && ( du[n,i] -= G[n,i]*noncoder*τ(noncoder, coder, α, β)*c(noncoder,coder, a=a) )           # non-prog death output due to cost
    (n < N && i < P) && ( du[n,i] += G[n+1,i]*(noncoder+1)*τ(noncoder+1, coder, α, β)*c(noncoder+1,coder, a=a) ) # non-prog death input due to cost
  end
end

function wrangle(sol)
  tot_out = []
  t_max = length(sol)-1
  for t=1:t_max
    out_num = zeros(41)
    out_denum = zeros(41)
    #!TODO: not just max but for all ts.
    for s=1:20
      for p=1:minimum([21,s+1])
        out_num[s+1] += ((p-1) / s) * sol[t][s-p+2,p]
        out_denum[s+1] += sol[t][s-p+2,p]
      end
    end
    push!(tot_out, out_num[2:20] ./ out_denum[2:20])
  end
  return tot_out
end

function plot_sol(sol, p; outdir=nothing)
  
  out = wrangle(sol)
  t_max = length(sol)-1
  param_lab = ["μ", "νₙ", "νₚ", "α", "β", "a", "p"]
  param_str = join(["$(pname)=$(p);" for (pname, p) in zip(param_lab, p)], ' ')

  ps=plot(2:20, out[2], label="t=2", legend=:outerright, top_margin = 20mm)
  for t=collect(5:5:30)
    plot!(2:20, out[t], label="t=$(t)") 
  end
  
  plot!(2:20, out[t_max], label="t=$(t_max)")
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

# plot_tryptic_cost(c, 1)

function main()
    args = parse_commandline()

    μ  = args["mu"]    # inflow new students-non coders
    νₙ = args["nu_n"]  # death rate non-coders
    νₚ = args["nu_p"]  # death rate coders
    α  = args["alpha"] # benefits non coders
    β  = args["beta"]  # benefits coders
    a  = args["a"]     # benefits coders
    
    println("initialize")
    u₀ = initialize_u0(N=20)
    params  = [μ, νₙ, νₚ, α, β, a]
    
    t_max = 4000
    tspan = (0., t_max)
    prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, params)
    println("solving problem")
    sol = solve(prob, Tsit5(), saveat=1, reltol=1e-8, abstol=1e-8)
    plot_sol(sol, params, outdir=args["o"])

    # @assert round(sum(sol[t_max]), digits= 2) == 1.0, "mat don't sum to 1.0"
end

main()

# prototyping ------------------------------------------------------------------

# μ  = 0.1   # inflow new students-non coders
# νₙ = 0.01    # death rate non-coders
# νₚ = 0.05    # death rate coders
# α  = 0.01    # benefits non coders
# β  = 0.02     # benefits coders
# a = 1       # parameter cost function
# params = [μ, νₙ, νₚ, α, β, a]

# u₀ = initialize_u0(N=20)

# t_max = 4000
# tspan = (0., t_max)

# prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, params)
# sol = solve(prob, Tsit5(), saveat=1, reltol=1e-8, abstol=1e-8)
# plot_sol(sol, params, outdir="figs")

# # checks
# round(sum(sol[1]), digits= 2)
# round(sum(sol[t_max]), digits= 2)

# round.(sum(sol[t_max], dims=1), digits=4)
# round.(sum(sol[t_max], dims=2), digits=4)
