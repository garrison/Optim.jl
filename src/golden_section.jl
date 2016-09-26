macro goldensectiontrace()
    quote
        if tracing
            dt = Dict()
            if extended_trace
                dt["current_minimizer"] = current_minimizer
                dt["x_lower"] = x_lower
                dt["x_upper"] = x_upper
            end
            update!(tr,
                    iteration,
                    minimum,
                    NaN,
                    dt,
                    store_trace,
                    show_trace,
                    show_every,
                    callback)
        end
    end
end

immutable GoldenSection <: Optimizer end

function optimize{T <: AbstractFloat}(f::Function, x_lower::T, x_upper::T,
                                      mo::GoldenSection;
                                      rel_tol::T = sqrt(eps(T)),
                                      abs_tol::T = eps(T),
                                      iterations::Integer = 1_000,
                                      store_trace::Bool = false,
                                      show_trace::Bool = false,
                                      callback = nothing,
                                      show_every = 1,
                                      extended_trace::Bool = false,
                                      nargs...)
    if !(x_lower < x_upper)
        error("x_lower must be less than x_upper")
    end

    # Save for later
    initial_lower = x_lower
    initial_upper = x_upper

    const golden_ratio::T = 0.5 * (3.0 - sqrt(5.0))

    current_minimizer = x_lower + golden_ratio*(x_upper-x_lower)
    minimum = f(current_minimizer)
    f_calls = 1 # Number of calls to f

    iteration = 0
    converged = false

    # Trace the history of states visited
    tr = OptimizationTrace{typeof(mo)}()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @goldensectiontrace

    while iteration < iterations

        tolx = rel_tol * abs(current_minimizer) + abs_tol

        x_midpoint = (x_upper+x_lower)/2

        if abs(current_minimizer - x_midpoint) <= 2*tolx - (x_upper-x_lower)/2
            converged = true
            break
        end

        iteration += 1

        if x_upper - current_minimizer > current_minimizer - x_lower
            x_new = current_minimizer + golden_ratio*(x_upper - current_minimizer)
            f_new = f(x_new)
            f_calls += 1
            if f_new < minimum
                x_lower = current_minimizer
                current_minimizer = x_new
                minimum = f_new
            else
                x_upper = x_new
            end
        else
            x_new = current_minimizer - golden_ratio*(current_minimizer - x_lower)
            f_new = f(x_new)
            f_calls += 1
            if f_new < minimum
                x_upper = current_minimizer
                current_minimizer = x_new
                minimum = f_new
            else
                x_lower = x_new
            end
        end

        @goldensectiontrace
    end

    return UnivariateOptimizationResults("Golden Section Search",
                                         initial_lower,
                                         initial_upper,
                                         current_minimizer,
                                         Float64(minimum),
                                         iteration,
                                         iteration == iterations,
                                         converged,
                                         rel_tol,
                                         abs_tol,
                                         tr,
                                         f_calls)
end
