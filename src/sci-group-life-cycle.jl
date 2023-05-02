using Pkg; Pkg.activate(".");
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots
using Plots.PlotMeasures

include("helpers.jl")


function initialize_u0(;N=40, M::Int=100)
  N_plus_one = N+1
  G = zeros(N_plus_one, N_plus_one)
  for _ in 1:100
    nb_non_prog = sum(rand(Binomial(1, .1), N_plus_one))
    nb_prog = sum(rand(Binomial(1, .05), N_plus_one))
    G[nb_non_prog+2, nb_prog+1] += 1
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

function frac_prog_in_groups(sol)
  """
  We want to know the (normalized) proportion of programmers by group size.

   - sol: list of matrix where sol[102][3,4] would be equal to the distribution
          at t=101 of groups with 2 programmers and 3 non-programmers.
  """
  N, P = size(first(sol)) # Sol is 21x21 but we have 20x20 people. 
                          # We have an extra col when #prog=0 or #non-prog=0
  
  S = (N-1) + (P-1) # max group size = #non-prog + # prog . But this is tricky.
                    # We'll limit the group size to be (S/2)+1 because when
                    # S/2 > 1, then we neccesarily start adding one or the other type of
                    # individuals, i.e. for gsize=40 necessarily you add 
                    # programmers or non-programmer after you hit 
                    # programmers=20 or non-programmers=20.
  
  t_max = length(sol)-1

  frac_prog_each_timestep = zeros(t_max, S÷2+1) # each vector is of 1x41, but first val should always=0 b/c we never have group_size=0
  
  for t=1:t_max
    nums =  zeros(S÷2+1)
    denums = zeros(S÷2+1)

    for gsize=1:(S÷2) # loop over group size 
      for p=1:gsize # loop over possible nb of programmers by group size.

        nb_prog=p-1     # idx_non_prog - 1 = #non-prog (b/c Julia is 1-based index) 
        n=gsize-p+2     # This  works because, say, we have gsize=2, (p=2|nb_prog=1). 
                        # We know if we have #prog=1, then #non-prog=1 (meaning its index n=2). 
                        # Thus 2 - 2 + 1 (#prog) + 1 (# we want index)
        nums[gsize] += (nb_prog / gsize) * sol[t][n,p]
        denums[gsize] += sol[t][n,p]
      end
    end
    frac_prog_each_timestep[t,:] = map(x -> isnan(x) ? 0. : float(x), nums ./ denums) # nans because 0/0. If it happens, we just put a zero.
  end
  return frac_prog_each_timestep
end

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

# μ  = 0.1   # inflow new students-non coders
# νₙ = 0.01    # death rate non-coders
# νₚ = 0.01    # death rate coders
# α  = 0.01    # benefits non coders
# β  = 0.1     # benefits coders
# a = 3.       # parameter cost function
# params = [μ, νₙ, νₚ, α, β, a]

# u₀ = initialize_u0(N=40)

# t_max = 4000
# tspan = (0., t_max)

# prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, params)
# sol = solve(prob, Rosenbrock23(), saveat=1, reltol=1e-6, abstol=1e-6)


# # checks
# round(sum(sol[1]), digits= 2)
# round(sum(sol[t_max]), digits= 2)

# round.(sum(sol[t_max], dims=1), digits=4)
# round.(sum(sol[t_max], dims=2), digits=4)


# --------------------------------- Analysis --------------------------------- #

# fraction of programmer by group size
# plot_sol(sol, params)  
# xlims!(0,40)
# plot!(size=(800,500))


# checking the nums and denums for plot_sol
function get_num_or_denum(sol; gsize, t, is_num=true)
  # gsize=25
  N, P = size(first(sol)) 
  out = []
  # equivalent to 1 if gsize <= (N-1)  else (gsize+1) - (N-1) 
  # for example, if gsize=25 => (25+1) - (21-1) = 6 (min programmers)
  #              if p=6 (or 5 prog), then we must have 20non-prog (or n=25-6+2=21).
  #              inversely, if p=21 (or 20 prog), then we must have 5 non-prog.
  min_p = gsize <= (N-1) ? 1 : (gsize+1) - (N-1) 
  for p=min_p:minimum([21, gsize+1])
    # p=15
    nb_prog = p-1
    n = gsize-p+2
    push!(out,  is_num ? (nb_prog / gsize) * sol[t][n,p] : sol[t][n,p] )
  end
  return out
end

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