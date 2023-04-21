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

# ╔═╡ d6b89c46-87e6-4376-8a5c-6f830f1b1159
using Plots, Distributions, StatsPlots, DataStructures, Graphs, MetaGraphs, GraphRecipes, LaTeXStrings, PlutoUI

# ╔═╡ 93033bee-03a7-453c-aef4-29e62e0465a8
md"""# Two ways to run a simulation"""

# ╔═╡ e7e5267a-a95c-4d4b-8b3a-b723de5e466e
function abm_sim(;p=0.5, tmax=25)
    histories = []
    nb_sims = 100
    
    for sims=1:nb_sims
        active_cells = 10
        history = []
        
        #for each generation
        for t=1:tmax
            #for each active cell
            for cell=1:active_cells
                #test division
                if rand() < p
                    active_cells += 2
                end
                #remove active cell either way
                active_cells -= 1
            #update history
            end
            history = push!(history, active_cells)
        end
        push!(histories, history)    
    end 
    return histories, tmax    
end

# ╔═╡ da97635f-4f97-4d77-b2c7-1e0c627421fe
function composition_sim(;p=0.5, tmax=25)
    histories = []
    nb_sims = 100

    for sims=1:nb_sims
        active_cells = 10
        history = []
        #for each generation
        for t=1:tmax
            #for each active cell
            #calculate the number of divisions
            divisions = rand(Binomial(active_cells, p))
            #remove active cells
            active_cells = 0
            #add new cells
            active_cells += 2*divisions
            #update history
            history = push!(history, active_cells)
        end
        push!(histories, history)    
    end 
    return histories, tmax
end

# ╔═╡ 3f8be61f-5513-4217-8572-843e4648d691
@bind p Slider(0.5:0.01:1.)

# ╔═╡ 3b4eaa39-6584-4ef2-8f26-e1533b5bb589
md"""p=$(p)"""

# ╔═╡ 17388c79-0376-45fd-863e-b4d6a65adb5b
md"""## Compare abm v. composition approach"""

# ╔═╡ 7aa55b39-9633-4fe9-a864-fa55c1925c8b
function timed_abm_sim(p=0.6, tmax=25)
    active_cells = ones(100)
    time_keeper = zeros(1)
    history = ones(1)
    #for each generation
    for t=1:tmax
        start = time()
        #for each simulation
        for sims=1:100
            for cell=1:(active_cells[sims])
                #test division
                if rand() < p
                    active_cells[sims] += 2
                end
                #remove active cell either way
                active_cells[sims] -= 1
            end
        end
        time_keeper = push!(time_keeper, (time() - start)/100)
        history = push!(history, mean(active_cells))
    end
    return history, time_keeper    
end

# ╔═╡ e72bb144-5762-4bde-87c7-2afeb46357e2
function timed_composition_sim(p=0.6, tmax=25)
    active_cells = ones(100)
    time_keeper = zeros(1)
    history = ones(1)
    #for each generation
    for t=1:tmax
        start = time()
        #for each simulation
        for sims=1:100
            #calculate the number of divisions
            divisions = rand(Binomial(active_cells[sims], p))
            #remove active cells
            active_cells[sims] = 0
            #add new cells
            active_cells[sims] += 2*divisions
        end
        #update time counter
        time_keeper = push!(time_keeper, (time() - start)/100)
        history = push!(history, mean(active_cells))
    end
    return history, time_keeper
end

# ╔═╡ d1b2eac7-adf5-492d-bb90-547c03645dee
md"""# Event-driven simulation"""

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
function run_sim(tmax)
    
    β = 0.0025 #transmission rate
    α = 0.01   #recovery rate
    I₀ = 0.01  #initial fraction of infected nodes
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

# ╔═╡ 8a258284-ed1c-4d06-ac61-e184a1cb4265
md"""## Plotting Helpers"""

# ╔═╡ 1d93cb54-8f6e-41de-a6d2-20f2b7a182ef
function plot_sim(histories, timestep)
    scatter(1:timestep, histories[1], color="black", alpha=.5, 
		    legend=true, label="Simulation")
    plot!(1:timestep, histories[2:end], color="black", alpha=0.25, label="")
    scatter!(1:timestep, histories[2:end], color="black", alpha=0.25, label="")
    ylabel!("Active cells")
    xlabel!("Generations")    
end

# ╔═╡ bfe096ed-8717-4287-9fc0-184e6d70dfcf
begin
	histories, timestep = abm_sim(p=p, tmax=25)
	plot_sim(histories, timestep)
end

# ╔═╡ ddce5a65-5f86-4a9e-8829-43fd04a5c16c
function plot_abm_composition_comp()
    history, time_keeper = timed_abm_sim();
    plot(history[2:end], time_keeper[2:end], color="blue", alpha=.5, label="")
    scatter!(history[2:end], time_keeper[2:end], color="blue", alpha=.5, label="")
    
    history, time_keeper = timed_composition_sim();
    plot!(history[2:end], time_keeper[2:end], color="orange", alpha=.5, label="")
    scatter!(history[2:end], time_keeper[2:end], color="orange", alpha=.5, label="")
    
    ylabel!("Seconds needed for generation")
    xlabel!("Average number of active cells per generation")    
end


# ╔═╡ 8da7d323-5a92-450a-97f1-d0cd978a675f
plot_abm_composition_comp()

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
	times, history = run_sim(500)
	plot_event_driven_sim(times, history)
end

# ╔═╡ Cell order:
# ╟─0dc28d42-dfb9-11ed-36db-31cab58a12ad
# ╠═d6b89c46-87e6-4376-8a5c-6f830f1b1159
# ╟─93033bee-03a7-453c-aef4-29e62e0465a8
# ╠═e7e5267a-a95c-4d4b-8b3a-b723de5e466e
# ╠═da97635f-4f97-4d77-b2c7-1e0c627421fe
# ╟─3b4eaa39-6584-4ef2-8f26-e1533b5bb589
# ╟─3f8be61f-5513-4217-8572-843e4648d691
# ╟─bfe096ed-8717-4287-9fc0-184e6d70dfcf
# ╟─17388c79-0376-45fd-863e-b4d6a65adb5b
# ╠═7aa55b39-9633-4fe9-a864-fa55c1925c8b
# ╠═e72bb144-5762-4bde-87c7-2afeb46357e2
# ╟─8da7d323-5a92-450a-97f1-d0cd978a675f
# ╟─d1b2eac7-adf5-492d-bb90-547c03645dee
# ╠═b508056b-b83c-4ef9-8b8c-2ef0d0ae1c92
# ╠═4927bc3f-84b2-4fbc-a7f3-f68615867f0d
# ╠═094be43d-e578-4034-8730-b027bb43548f
# ╠═608bef71-4ce0-4c9f-8ff5-29f96c580c2d
# ╟─8a258284-ed1c-4d06-ac61-e184a1cb4265
# ╟─1d93cb54-8f6e-41de-a6d2-20f2b7a182ef
# ╟─ddce5a65-5f86-4a9e-8829-43fd04a5c16c
# ╟─b84eee8d-82b0-4fcf-8bd4-05ed66b0db22
