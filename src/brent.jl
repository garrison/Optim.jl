macro brenttrace()
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

immutable Brent <: Optimizer end

function optimize{T <: AbstractFloat}(
        f::Function, x_lower::T, x_upper::T,
        mo::Brent;
        rel_tol::T = sqrt(eps(T)),
        abs_tol::T = eps(T),
        iterations::Integer = 1_000,
        store_trace::Bool = false,
        show_trace::Bool = false,
        callback = nothing,
        show_every = 1,
        extended_trace::Bool = false)

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

    step = zero(T)
    step_old = zero(T)

    current_minimizer_old = current_minimizer
    current_minimizer_old_old = current_minimizer

    minimum_old = minimum
    minimum_old_old = minimum

    iteration = 0
    converged = false

    # Trace the history of states visited
    tr = OptimizationTrace{typeof(mo)}()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    @brenttrace

    while iteration < iterations

        p = zero(T)
        q = zero(T)

        tolx = rel_tol * abs(current_minimizer) + abs_tol

        x_midpoint = (x_upper+x_lower)/2

        if abs(current_minimizer - x_midpoint) <= 2*tolx - (x_upper-x_lower)/2
            converged = true
            break
        end

        iteration += 1

        if abs(step_old) > tolx
            # Compute parabola interpolation
            # current_minimizer + p/q is the optimum of the parabola
            # Also, q is guaranteed to be positive

            r = (current_minimizer - current_minimizer_old) * (minimum - minimum_old_old)
            q = (current_minimizer - current_minimizer_old_old) * (minimum - minimum_old)
            p = (current_minimizer - current_minimizer_old_old) * q - (current_minimizer - current_minimizer_old) * r
            q = 2(q - r)

            if q > 0
                p = -p
            else
                q = -q
            end
        end

        if abs(p) < abs(q*step_old/2) && p < q*(x_upper-current_minimizer) && p < q*(current_minimizer-x_lower)
            step_old = step
            step = p/q

            # The function must not be evaluated too close to x_upper or x_lower
            x_temp = current_minimizer + step
            if ((x_temp - x_lower) < 2*tolx || (x_upper - x_temp) < 2*tolx)
                step = (current_minimizer < x_midpoint) ? tolx : -tolx
            end
        else
            step_old = (current_minimizer < x_midpoint) ? x_upper - current_minimizer : x_lower - current_minimizer
            step = golden_ratio * step_old
        end

        # The function must not be evaluated too close to current_minimizer
        if abs(step) >= tolx
            x_new = current_minimizer + step
        else
            x_new = current_minimizer + ((step > 0) ? tolx : -tolx)
        end

        f_new = f(x_new)
        f_calls += 1

        if f_new <= minimum
            if x_new < current_minimizer
                x_upper = current_minimizer
            else
                x_lower = current_minimizer
            end
            current_minimizer_old_old = current_minimizer_old
            minimum_old_old = minimum_old
            current_minimizer_old = current_minimizer
            minimum_old = minimum
            current_minimizer = x_new
            minimum = f_new
        else
            if x_new < current_minimizer
                x_lower = x_new
            else
                x_upper = x_new
            end
            if f_new <= minimum_old || current_minimizer_old == current_minimizer
                current_minimizer_old_old = current_minimizer_old
                minimum_old_old = minimum_old
                current_minimizer_old = x_new
                minimum_old = f_new
            elseif f_new <= minimum_old_old || current_minimizer_old_old == current_minimizer || current_minimizer_old_old == current_minimizer_old
                current_minimizer_old_old = x_new
                minimum_old_old = f_new
            end
        end

        @brenttrace
    end

    return UnivariateOptimizationResults("Brent's Method",
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
