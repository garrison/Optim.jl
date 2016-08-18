log_temperature(t::Real) = 1 / log(t)

constant_temperature(t::Real) = 1.0

function default_neighbor!(x::Array, x_proposal::Array)
    @assert size(x) == size(x_proposal)
    for i in 1:length(x)
        @inbounds x_proposal[i] = x[i] + randn()
    end
    return
end

immutable SimulatedAnnealing <: Optimizer
    neighbor!::Function
    temperature::Function
    keep_best::Bool # not used!?
end

SimulatedAnnealing(; neighbor!::Function = default_neighbor!,
                     temperature::Function = log_temperature,
                     keep_best::Bool = true) =
  SimulatedAnnealing(neighbor!, temperature, keep_best)

method_string(method::SimulatedAnnealing) = "Simulated Annealing"

type SimulatedAnnealingState{T}
  n::Int64
  iteration::Int64
  x_current::Array{T}
  x_proposal
  x
  f_x
  f_x_current::T
  f_proposal::T
  f_calls::Int64
  g_calls::Int64
end

initialize_state(method::SimulatedAnnealing, options, d, initial_x::Array) = initialize_state(method, options, d.f, initial_x)

function initialize_state{T}(method::SimulatedAnnealing, options, f::Function, initial_x::Array{T})
    # Count number of parameters
    n = length(initial_x)

    # Store f(x) in f_x
    f_x = f(initial_x)
    f_calls = 1

    # Store the best state ever visited
    best_x = copy(initial_x)
    best_f_x = f_x
    iteration = 1
    SimulatedAnnealingState(n, 1, copy(initial_x), copy(initial_x), best_x, best_f_x, f_x, f_x, f_calls, 0)
end

update!(d, state::SimulatedAnnealingState, method::SimulatedAnnealing) = update!(d.f, state, method)
function update!{T}(f::Function, state::SimulatedAnnealingState{T}, method::SimulatedAnnealing)

    # Determine the temperature for current iteration
    t = method.temperature(state.iteration)

    # Randomly generate a neighbor of our current state
    method.neighbor!(state.x_current, state.x_proposal)

    # Evaluate the cost function at the proposed state
    state.f_proposal = f(state.x_proposal)
    state.f_calls += 1

    if state.f_proposal <= state.f_x_current
        # If proposal is superior, we always move to it
        copy!(state.x_current, state.x_proposal)
        state.f_x_current = state.f_proposal

        # If the new state is the best state yet, keep a record of it
        if state.f_proposal < state.f_x
            state.f_x = state.f_proposal
            copy!(state.x, state.x_proposal)
        end
    else
        # If proposal is inferior, we move to it with probability p
        p = exp(-(state.f_proposal - state.f_x_current) / t)
        if rand() <= p
            copy!(state.x_current, state.x_proposal)
            state.f_x_current = state.f_proposal
        end
    end

    state.iteration += 1
    false
end
