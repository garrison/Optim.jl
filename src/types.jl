abstract Optimizer
immutable OptimizationOptions{TCallback <: Union{Void, Function}}
    x_tol::Float64
    f_tol::Float64
    g_tol::Float64
    iterations::Int
    store_trace::Bool
    show_trace::Bool
    extended_trace::Bool
    autodiff::Bool
    show_every::Int
    callback::TCallback
    time_limit::Float64
end

function OptimizationOptions(;
        x_tol::Real = 1e-32,
        f_tol::Real = 1e-32,
        g_tol::Real = 1e-8,
        iterations::Integer = 1_000,
        store_trace::Bool = false,
        show_trace::Bool = false,
        extended_trace::Bool = false,
        autodiff::Bool = false,
        show_every::Integer = 1,
        callback = nothing,
        time_limit = NaN)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    OptimizationOptions{typeof(callback)}(
        Float64(x_tol), Float64(f_tol), Float64(g_tol), Int(iterations),
        store_trace, show_trace, extended_trace, autodiff, Int(show_every),
        callback, time_limit)
end

function print_header(options::OptimizationOptions)
    if options.show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
end

function print_header(method::Optimizer)
        @printf "Iter     Function value   Gradient norm \n"
end

immutable OptimizationState{T <: Optimizer}
    iteration::Int
    value::Float64
    g_norm::Float64
    metadata::Dict
end

typealias OptimizationTrace{T} Vector{OptimizationState{T}}

abstract OptimizationResults

type MultivariateOptimizationResults{T,N,M} <: OptimizationResults
    method::String
    initial_x::Array{T,N}
    minimizer::Array{T,N}
    minimum::Float64
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::Float64
    f_converged::Bool
    f_tol::Float64
    g_converged::Bool
    g_tol::Float64
    trace::OptimizationTrace{M}
    f_calls::Int
    g_calls::Int
    h_calls::Int
end

type UnivariateOptimizationResults{T,M} <: OptimizationResults
    method::String
    initial_lower::T
    initial_upper::T
    minimizer::T
    minimum::Float64
    iterations::Int
    iteration_converged::Bool
    converged::Bool
    rel_tol::Float64
    abs_tol::Float64
    trace::OptimizationTrace{M}
    f_calls::Int
end

immutable NonDifferentiableFunction
    f::Function
end

immutable DifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
end

immutable TwiceDifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
    h!::Function
end

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.g_norm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

function Base.show(io::IO, tr::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in tr
        show(io, state)
    end
    return
end

function Base.show(io::IO, r::MultivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" method(r)
    if length(join(initial_state(r), ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(initial_state(r), ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(initial_state(r)[1:2], ",")
    end
    if length(join(minimizer(r), ",")) < 40
        @printf io " * Minimizer: [%s]\n" join(minimizer(r), ",")
    else
        @printf io " * Minimizer: [%s, ...]\n" join(minimizer(r)[1:2], ",")
    end
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: %s\n" converged(r)
    if r.method == "Nelder-Mead"
        @printf io "   *  √(Σ(yᵢ-ȳ)²)/n < %.1e: %s\n" g_tol(r) g_converged(r)
    else
        @printf io "   * |x - x'| < %.1e: %s\n" x_tol(r) x_converged(r)
        @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" f_tol(r) f_converged(r)
        @printf io "   * |g(x)| < %.1e: %s\n" g_tol(r) g_converged(r)
    end
    @printf io "   * Reached Maximum Number of Iterations: %s\n" iteration_limit_reached(r)
    @printf io " * Objective Function Calls: %d\n" f_calls(r)
    if !(r.method in ("Nelder-Mead", "Simulated Annealing"))
        @printf io " * Gradient Calls: %d" g_calls(r)
    end
    return
end

function Base.show(io::IO, r::UnivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" method(r)
    @printf io " * Search Interval: [%f, %f]\n" lower_bound(r) upper_bound(r)
    @printf io " * Minimizer: %e\n" minimizer(r)
    @printf io " * Minimum: %e\n" minimum(r)
    @printf io " * Iterations: %d\n" iterations(r)
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" rel_tol(r) abs_tol(r) converged(r)
    @printf io " * Objective Function Calls: %d" f_calls(r)
    return
end

function Base.append!(a::MultivariateOptimizationResults, b::MultivariateOptimizationResults)
    a.iterations += iterations(b)
    a.minimizer = minimizer(b)
    a.minimum = minimum(b)
    a.iteration_converged = iteration_limit_reached(b)
    a.x_converged = x_converged(b)
    a.f_converged = f_converged(b)
    a.g_converged = g_converged(b)
    append!(a.trace, b.trace)
    a.f_calls += f_calls(b)
    a.g_calls += g_calls(b)
end

# TODO: Expose ability to do forward and backward differencing
function DifferentiableFunction(f::Function)
    function g!(x::Array, storage::Array)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!)
end

function DifferentiableFunction(f::Function, g!::Function)
    function fg!(x::Array, storage::Array)
        g!(x, storage)
        return f(x)
    end
    return DifferentiableFunction(f, g!, fg!)
end

function TwiceDifferentiableFunction(f::Function)
    function g!(x::Vector, storage::Vector)
        Calculus.finite_difference!(f, x, storage, :central)
        return
    end
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    function h!(x::Vector, storage::Matrix)
        Calculus.finite_difference_hessian!(f, x, storage)
        return
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!)
end

function TwiceDifferentiableFunction(f::Function,
                                     g!::Function,
                                     h!::Function)
    function fg!(x::Vector, storage::Vector)
        g!(x, storage)
        return f(x)
    end
    return TwiceDifferentiableFunction(f, g!, fg!, h!)
end

# A cache for results from line search methods (to avoid recomputation)
type LineSearchResults{T}
    alpha::Vector{T}
    value::Vector{T}
    slope::Vector{T}
    nfailures::Int
end

LineSearchResults{T}(::Type{T}) = LineSearchResults(T[], T[], T[], 0)

Base.length(lsr::LineSearchResults) = length(lsr.alpha)

function Base.push!{T}(lsr::LineSearchResults{T}, a::T, v::T, d::T)
    push!(lsr.alpha, a)
    push!(lsr.value, v)
    push!(lsr.slope, d)
    return
end

function clear!(lsr::LineSearchResults)
    empty!(lsr.alpha)
    empty!(lsr.value)
    empty!(lsr.slope)
    return
    # nfailures is deliberately not set to 0
end
