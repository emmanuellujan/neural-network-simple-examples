"""

Fitting a second order differential equation using a physical informed neural network

"""

using LinearAlgebra
using GalacticOptim, Optim, Flux
using Plots


"""
Finite difference derivatives
"""
function deriv(f, t)
    dt = 0.0001
    return (f(t+dt) - f(t-dt)) / (2dt)
end
function deriv2(f, t)
    dt = 0.0001
    return (f(t+dt) - 2f(t) + f(t-dt)) / dt^2
end

"""
Target function
See https://ocw.mit.edu/courses/mathematics/18-03sc-differential-equations-fall-2011/unit-ii-second-order-constant-coefficient-linear-equations/exam-2/MIT18_03SCF11_ex2.pdf
"""
function target(t)
    return 0.5 * t * sin(2t)
end

"""
Activation function
See https://en.wikipedia.org/wiki/Activation_function
"""
function σ(x) 
    return 1.0 / ( 1.0 + exp(-x) )
end

"""
Neural network function
See: Universal Approximation Bounds for Superpositions of a Sigmoidal Function.
     Andrew R. Barron. IEEE TRANSACTIONS ON INFORMATI0N THEORY, VOL. 39, NO.3, MAY 1993
"""
function nn(t, w, p) 
    nnl = p[1]
    w_hl = w[1:nnl]; b_hl = w[nnl+1:2nnl]
    w_out = w[2nnl+1:3nnl]; b_out = w[3nnl+1]
    hl = σ.(t * w_hl .+ b_hl) 
    return sum(hl .* w_out) + b_out
end

"""
Loss function
"""
function loss(w, p)
    m = 2.0; b = 0.0; k = 8.0
    nnw = (t -> nn(t, w, p))
    residue(t) = m * deriv2(nnw, t) + b * deriv(nnw, t) + k * nnw(t) - 4cos(2t)
    residue_sum = sum([ residue(t)^2 for t in range ]) / length(range)
    return (nnw(0.0) - target(0.0))^2 +
           (deriv(nnw, 0.0) - deriv(target, 0.0))^2 +
           (deriv2(nnw, 0.0) - deriv2(target, 0.0))^2 +
           residue_sum
end

# Optimization
range = 0.0:0.01:5.0
nhl = 10; nw = 3*nhl+1; w0 = 30.0*rand(nw).-15.0; p = [nhl]
f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
prob = OptimizationProblem(f, w0, p)
sol = solve(prob, ADAM(0.1), maxiters=4000)
prob = remake(prob, u0=sol.minimizer)
sol = solve(prob, ADAM(0.001), maxiters=2000)

# Plot result
plot([ target(t) for t in range ])
plot!([ nn(t, sol, p) for t in range ])






