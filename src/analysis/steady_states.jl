using Pkg; Pkg.activate(".");
using ArgParse, Distributions, StatsBase, OrdinaryDiffEq, Plots
using Plots.PlotMeasures
using OrderedCollections

include("../helpers.jl")
# include("../sci-group-life-cycle.jl")

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
 sol => list of matrix where sol[102][3,4] would be equal to the dist.
       at t=101 of groups with 2 progs and 3 non-progs.
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
        # t, gsize, p = t_max, 10, 10
        nb_prog=p-1     # idx_non_prog - 1 = #non-prog (b/c Julia is 1-based index) 
        n=gsize-p+2     # This  works because, say, we have gsize=2, (p=2|nb_prog=1). 
                        # We know if we have #prog=1, then #non-prog=1 (meaning its index n=2). 
                        # Thus 2 - 2 + 1 (#prog) + 1 (# we want index)
        nums[gsize] += (nb_prog / gsize) * sol[t][n,p]
        denums[gsize] += sol[t][n,p]
        (nb_prog / gsize) * sol[t][n,p] / sol[t][n,p]
      end
    end
    frac_prog_each_timestep[t,:] = map(x -> isnan(x) ? 0. : float(x), nums ./ denums) # nans because 0/0. If it happens, we just put a zero.
  end
  return frac_prog_each_timestep
end
  
function run_sci_group(p_lookup)
    u₀ = initialize_u0(N=20)
    prob = ODEProblem(life_cycle_research_groups!, u₀, tspan, collect(values(p_lookup)))
    sol = solve(prob, Rosenbrock23(), saveat=1, reltol=1e-6, abstol=1e-6)
    return sol, frac_prog_in_groups(sol);
end

# target_p, first_val = "β", 0.05
tspan = (0., 4000) 

μ  = .1
νₙ = .01 
νₚ = .01
α  = .01 
β  = .76
a  = 3

params = [μ, νₙ, νₚ, α, β, a ]
param_lab = ["μ", "νₙ", "νₚ", "α", "β", "a"]
param_lookup = OrderedDict(pname => p for (pname, p) in zip(param_lab, params))

sol, out = run_sci_group(param_lookup)
t_max, last_t = length(sol)-1, length(out[1,:])-1
sol[t_max]

plot(1:last_t, out[t_max,1:last_t], label="β=$(β)")

μ  = .01
νₙ = .01 
νₚ = .5
α  = .01 
β  = .9
a  = 5
params = [μ, νₙ, νₚ, α, β, a ]
param_lookup = OrderedDict(pname => p for (pname, p) in zip(param_lab, params))

sol, out = run_sci_group(param_lookup)
t_max, last_t = length(sol)-1, length(out[1,:])-1
plot!(1:last_t, out[t_max,1:last_t], label="β=$(β)")

xlabel!("group size")
ylabel!("proportion programmers")
ylims!(0,1)

# for new_p=.1:.33:1.
#     println(new_p)
#     param_lookup["$(target_p)"] = new_p
#     sol, out = run_sci_group(param_lookup)
#     t_max, last_t = length(sol)-1, length(out[1,:])-1
#     plot!(1:last_t, out[t_max,1:last_t], label="$(target_p)=$(new_p)")
# end


frac_prog_each_timestep[t_max,1:last_t]
out[t_max,1:last_t]