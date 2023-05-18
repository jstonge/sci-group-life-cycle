### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0dc28d42-dfb9-11ed-36db-31cab58a12ad
using Pkg; Pkg.activate("..")

# ╔═╡ d6b89c46-87e6-4376-8a5c-6f830f1b1159
using Plots, Distributions, StatsPlots, DataStructures, Graphs, MetaGraphs, GraphRecipes, LaTeXStrings, PlutoUI

# ╔═╡ d1b2eac7-adf5-492d-bb90-547c03645dee
md"""# Event-driven simulation"""

# ╔═╡ 5f77ed94-d2c1-40dd-b4d5-6c00c54a2b3b


# ╔═╡ b508056b-b83c-4ef9-8b8c-2ef0d0ae1c92
function infect_new_node(event_queue, G, node, beta, alpha, t)
    # Set Recovery period
    # The time to a Poisson event 
    #     follows an exponential distribution with average 1/rate
    tau_r = first(rand(Exponential(1/alpha), 1))
    # Add the recovery to the queue which orders by time, 
    # but also make sure to save information needed: node and event type.
    
    push!(event_queue, (t+tau_r, node, "recovery"))

    # Set infection times
    
    for neighbor in neighbors(G, node) #loop over ALL neighbors
        tau_i = first(rand(Exponential(1/beta), 1))
        if tau_i < tau_r
            push!(event_queue, (t+tau_i, neighbor, "infection"))
        end
    end

    return event_queue
end

# ╔═╡ 4927bc3f-84b2-4fbc-a7f3-f68615867f0d
function init_event_driven_sim(p; n_nodes=1000)
    β, α, I₀ = p
    # Create a network
    G = MetaGraph(barabasi_albert(n_nodes, 2))
    
    # Event queue for continuous time events
    event_queue = BinaryMinHeap{Any}()
    
    # Initial conditions
    t = 0
    I = 0
    
    for node in vertices(G) #loop over nodes
        set_prop!(G, node, :state, "susceptible")
        if rand() < I₀
            set_prop!(G, node, :state, "infected")
            event_queue = infect_new_node(event_queue, G, node, β, α, t)
            I += 1
        end
    end

    return G, event_queue, I
end

# ╔═╡ 094be43d-e578-4034-8730-b027bb43548f
function run_sim(p; tmax)
    β, α, I₀ = p
    t = 0
    params = (β, α, I₀)
    G, event_queue, I = init_event_driven_sim(params)

    history = []
    times = []
    push!(history, I/nv(G))
    push!(times, t)
    
    while t<tmax
        time, node, event = pop!(event_queue)
        
        if event == "recovery" && props(G,node)[:state] == "infected"
            set_prop!(G, node, :state, "recovered")
            I -= 1
        end
        if event == "infection" && props(G,node)[:state] == "susceptible"
            set_prop!(G, node, :state, "infected")
            event_queue = infect_new_node(event_queue, G, node, β, α, t)
            I += 1
        end
        
        t=time
        push!(history, I/nv(G))
        push!(times, t)
    end
    return times, history
end

# ╔═╡ eb01ae47-9a2f-4603-8ee6-ef9894189d36
md"""
β = $(@bind β Slider(.0015:0.0005:.0035))
α = $(@bind α Slider(0.005:0.001:.02))
I₀ = $(@bind I₀ Slider(0.005:0.001:.02))
"""

# ╔═╡ 34c40cbc-ab0c-49df-930b-c59678cc5013
println("β=$(β); α=$(α); I₀=$(I₀)")

# ╔═╡ 8a258284-ed1c-4d06-ac61-e184a1cb4265
md"""## Plotting Helpers"""

# ╔═╡ b84eee8d-82b0-4fcf-8bd4-05ed66b0db22
function plot_event_driven_sim(times, history)
    
    #Add null simulation with label in legend and label the axes
    plot(times, history, color="black", alpha=0.25, label="")
    scatter!(times, history, color="black", alpha=0.25, label="Simulations")
    ylabel!("Infected nodes")
    xlabel!("Time")
end

# ╔═╡ 608bef71-4ce0-4c9f-8ff5-29f96c580c2d
begin
	params = (β, α, I₀)
	times, history = run_sim(params; tmax=500)
	plot_event_driven_sim(times, history)
end

# ╔═╡ Cell order:
# ╟─0dc28d42-dfb9-11ed-36db-31cab58a12ad
# ╠═d6b89c46-87e6-4376-8a5c-6f830f1b1159
# ╟─d1b2eac7-adf5-492d-bb90-547c03645dee
# ╠═5f77ed94-d2c1-40dd-b4d5-6c00c54a2b3b
# ╠═b508056b-b83c-4ef9-8b8c-2ef0d0ae1c92
# ╠═4927bc3f-84b2-4fbc-a7f3-f68615867f0d
# ╠═094be43d-e578-4034-8730-b027bb43548f
# ╟─eb01ae47-9a2f-4603-8ee6-ef9894189d36
# ╟─608bef71-4ce0-4c9f-8ff5-29f96c580c2d
# ╟─34c40cbc-ab0c-49df-930b-c59678cc5013
# ╟─8a258284-ed1c-4d06-ac61-e184a1cb4265
# ╠═b84eee8d-82b0-4fcf-8bd4-05ed66b0db22
