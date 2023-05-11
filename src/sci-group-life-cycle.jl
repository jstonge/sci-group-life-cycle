using Pkg; Pkg.activate(".")
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots, OffsetArrays
using Plots.PlotMeasures

using CSV
using DataFrames


include("helpers.jl")

τ(n, i, α, β; b=.9) = exp(-α + β*(1 - c(n, i, b=b))) # group benefits
c(n, i; a=3, b=0.9) = n == i == 0 ? b : b * exp(-a*i / n)  # cost function
σ(n, i; K=25, μ=0.1)  = μ*(i+n+1) * (1-(i+n+1)/K)

function initialize_u0(;N=40, M::Int=100)
  G = Float64.(reshape(zeros(N+1,N+1), N+1, N+1))
  G = OffsetArray(G, 0:N, 0:N)
  for group in 1:M
    gsize = sum(rand(Binomial(1, .1), N))
    nb_prog = sum(rand(Binomial(1, .1), gsize))
    nb_non_prog = gsize - nb_prog
    G[nb_non_prog, nb_prog] += 1
  end
  G = G ./ M
  return G
end

function wrangle_Ig(sol, only_prog=true)
  #only_prog=true
  N, P = size(first(sol))
  max_group = only_prog ? P : (P+N-1)
  
  sol_wrangled = zeros((length(sol), max_group))
  for t=eachindex(sol)
    # t=10
    num = OffsetVector(zeros(max_group), 0:max_group-1)
    denum = OffsetVector(zeros(max_group), 0:max_group-1)
        
    for gsize=0:(max_group-1)
      min_range = gsize >= P ? (gsize - P + 1) : 0
      for p=min_range:minimum([P-1, gsize])
            n=gsize-p
            # println((p,n,gsize))
            if only_prog == true # we only do numerator
                num[gsize] += p==gsize==0 ? 0 : (p / gsize) * sol[t][p, n]
            end
            denum[gsize] += sol[t][p, n]
          end
      end

      if only_prog == true
        weighted_sum  = map(x -> isnan(x) ? 0. : float(x), num ./ denum)
        sol_wrangled[t,:] = weighted_sum / sum(weighted_sum)
      else # fraction of group is just the denum
          sol_wrangled[t,:] = denum
      end
    end

  return sol_wrangled
end

function life_cycle_research_groups!(du, u, p, t)
  N, P = size(u) 
  G = u # G for groups
  μ, νₙ, νₚ, α, β, b, K = p

  for i=0:(P-1), n=0:(N-1)

    du[i,n] = 0
    
    n > 0 && ( du[i,n] += G[i,n-1]*σ(n-1, i, K=K, μ=μ) ) # non-prog repro input
    (n+1) < N && ( du[i,n] -= G[i,n]*σ(n, i, K=K, μ=μ) ) # non-prog repro output

    (n+1) < N && ( du[i,n] += G[i,n+1]*νₙ*(n+1) ) # non-prog death input
    du[i,n] -= G[i,n]*n*νₙ # non-prog death output
    
    (i+1) < P && ( du[i,n] += G[i+1,n]*νₚ*(i+1) ) # prog death input
    du[i,n] -= G[i,n]*i*νₚ # prog death output
    
    (i > 0 && (n+1) < N) && ( du[i,n] += G[i-1,n+1]*(n+1)*τ(n+1, i-1, α, β, b=b)*(1-c(n+1,i-1, b=b)) ) # non-prog to prog input
    (i+1) < P && ( du[i,n] -= G[i,n]*n*τ(n, i, α, β, b=b)*(1-c(n, i, b=b)) ) # non-prog to prog output
        
    (i+1) < P && ( du[i,n] -= G[i,n]*n*τ(n, i, α, β, b=b)*c(n,i,b=b) ) # non-prog death output due to cost
    ((n+1) < N && (i+1) < P) && ( du[i,n] += G[i,n+1]*(n+1)*τ(n+1, i, α, β, b=b)*c(n+1,i,b=b) ) # non-prog death input due to cost
  end
end



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

# main()

# prototyping ------------------------------------------------------------------

# plot_tryptic_cost(c, 1)

μ  = 0.5   # inflow new students-non coders
νₙ = 0.01    # death rate non-coders
νₚ = 0.01    # death rate coders
α  = 0.01    # benefits non coders
β  = 0.1     # benefits coders
# a = 3.       # parameter cost function
b = .5
K = 40
params = (0.5, 0.01, 0.01, 0.01, 0.1, .5, 40)
# params = [μ, νₙ, νₚ, α, β, b, K]

u₀ = initialize_u0(N=40, M=1000)

t_max = 50
tspan = (0., t_max)

prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, params)
sol = solve(prob, Rosenbrock23(), saveat=1, reltol=1e-6, abstol=1e-6)

# checks
sol[t_max]
round.(sol[t_max], digits=2)

round(sum(sol[1]), digits= 2)
round(sum(sol[t_max]), digits= 2)

round.(sum(sol[t_max], dims=1), digits=4)
round.(sum(sol[t_max], dims=2), digits=4)

round.(sol[t_max][21,:], digits=4)


# sol_Wrangled = frac_prog_in_groups(sol)
sol_wrangled = wrangle_Ig(sol, true)
sol_wrangled_group = wrangle_Ig(sol, false)


CSV.write("src/test.csv", DataFrame(sol_wrangled_group, :auto))

@assert (sum(sol_wrangled[t_max,:]) > .998) && (sum(sol_wrangled[t_max,:]) < 1.001) "Should always be normalized"
@assert (sum(sol_wrangled_group[t_max,:]) > .998) && (sum(sol_wrangled_group[t_max,:]) < 1.001) == 1.0 "Should always be normalized"


# sol_wrangled = frac_prog_in_groups(sol)

ss = [_ for _ in sol_wrangled[t_max,:]]
top_ind = [i for (i, val) in enumerate(sol_wrangled[t_max,:]) if val > 0.1]

for gsize=1:40
  if gsize == 1
    ps=plot([sol_wrangled[t,1] for t=1:t_max], color="black", label="gsize=$(gsize-1)")
    scatter!([sol_wrangled[t,1] for t=1:t_max], color="black", label="")
  else
    if gsize ∈ top_ind
      plot!([sol_wrangled[t,gsize] for t=1:t_max], color="red", label="gsize=$(gsize-1)")
      scatter!([sol_wrangled[t,gsize] for t=1:t_max], color="red", label="")
    else 
      plot!([sol_wrangled[t,gsize] for t=1:t_max], color="grey", alpha=0.4, label="")
      scatter!([sol_wrangled[t,gsize] for t=1:t_max], color="grey", alpha=0.4, label="")
    end
  end
end

# ylims!(0,1)
xlabel!("time")
ylabel!("frac programmer")



ss = [_ for _ in sol_wrangled_group[t_max,:]]
top_ind = [i for (i, val) in enumerate(sol_wrangled_group[t_max,:]) if val > 0.01]

for gsize=1:size(sol_wrangled_group, 2)
  if gsize == 1
    ps=plot([sol_wrangled_group[t,1] for t=1:t_max], color="black", label="")
    # scatter!([sol_wrangled_group[t,1] for t=1:t_max], color="black", label="")
  else
    if gsize ∈ top_ind
      plot!([sol_wrangled_group[t,gsize] for t=1:t_max], label="gsize=$(gsize-1)")
      # scatter!([sol_wrangled_group[t,gsize] for t=1:t_max], color="black", label="")
    else 
      plot!([sol_wrangled_group[t,gsize] for t=1:t_max], color="grey", alpha=0.4, label="")
      # scatter!([sol_wrangled_group[t,gsize] for t=1:t_max], color="grey", alpha=0.4, label="")
    end
  end
end

# ylims!(0,1)
xlabel!("time")
ylabel!("frac group size")
savefig("julia_ames.png")

# debugging -------------------------------------------------------------------


# checking the nums and denums for plot_sol
# function get_num_or_denum(sol; gsize, t, is_num=true)
#   # gsize=25
#   N, P = size(first(sol)) 
#   out = []
#   # equivalent to 1 if gsize <= (N-1)  else (gsize+1) - (N-1) 
#   # for example, if gsize=25 => (25+1) - (21-1) = 6 (min programmers)
#   #              if p=6 (or 5 prog), then we must have 20non-prog (or n=25-6+2=21).
#   #              inversely, if p=21 (or 20 prog), then we must have 5 non-prog.
#   min_p = gsize <= (N-1) ? 1 : (gsize+1) - (N-1) 
#   for p=min_p:minimum([21, gsize+1])
#     # p=15
#     nb_prog = p-1
#     n = gsize-p+2
#     push!(out,  is_num ? (nb_prog / gsize) * sol[t][n,p] : sol[t][n,p] )
#   end
#   return out
# end

# num = get_num_or_denum(sol, gsize=10, t=t_max, is_num=true)
# denum = get_num_or_denum(sol, gsize=10, t=t_max, is_num=false)
# "$(sum(num)) / $(sum(denum)) = $(sum(num) / sum(denum))"

# num = get_num_or_denum(sol, gsize=25, t=t_max, is_num=true)
# denum = get_num_or_denum(sol, gsize=25, t=t_max, is_num=false)
# "$(sum(num)) / $(sum(denum)) = $(sum(num) / sum(denum))"
# sol[t_max]

# num = get_num_or_denum(sol, gsize=32, t=t_max, is_num=true)
# denum = get_num_or_denum(sol, gsize=32, t=t_max, is_num=false)
# "$(sum(num)) / $(sum(denum)) = $(sum(num) / sum(denum))"


