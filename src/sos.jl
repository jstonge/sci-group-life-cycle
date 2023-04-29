using Pkg; Pkg.activate(".")
using Plots, Distributions, StatsPlots, DataStructures, Graphs, MetaGraphs, GraphRecipes, LaTeXStrings


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