using Pkg; Pkg.activate(".")
using Plots, Distributions, StatsPlots, DataStructures, Graphs, MetaGraphs, GraphRecipes, LaTeXStrings

function plot_sim(histories, timestep)
    scatter(1:timestep, histories[1], color="black", alpha=.5, legend=true, label="Simulation")
    plot!(1:timestep, histories[2:end], color="black", alpha=0.25, label="")
    scatter!(1:timestep, histories[2:end], color="black", alpha=0.25, label="")
    ylabel!("Active cells")
    xlabel!("Generations")    
end

function abm_sim(p=0.5, tmax=25)
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

function composition_sim(p=0.5, tmax=25)
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

histories, timestep = abm_sim()
plot_sim(histories, timestep)

histories, timestep = composition_sim()
plot_sim(histories, timestep)


function timed_abm_sim(p=0.6, tmax=15)
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

function timed_composition_sim(p=0.6, tmax=15)
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

plot_abm_composition_comp()

function plot_graph_sir()
    graphplot(G, nodeshape=:circle, color=:black, nodesize=0.2, names = 1:100, curves=false)
    plot!(size=(1000,1000))    
end

function plot_assumptions(β=0.01, α=0.025)
    plot(Exponential(1/β), label="transmision rate = " * L"\frac{1}{\beta=0.01}" * " = $(1/β)")
    plot!(Exponential(1/α), label="recovery rate = " * L"\frac{1}{\alpha=0.025}" * " = $(1/α)")
    ylabel!("p(x)")
    xlabel!("time")
end

plot_assumptions()

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

times, history = run_sim(500)

function plot_event_driven_sim(times, history)
    
    #Add null simulation with label in legend and label the axes
    plot(times, history, color="black", alpha=0.25, label="")
    scatter!(times, history, color="black", alpha=0.25, label="Simulations")
    ylabel!("Infected nodes")
    xlabel!("Time")
end

plot_event_driven_sim(times, history)