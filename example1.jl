using LinearAlgebra
using GalacticOptim, Optim, Flux
using Plots

"""
Target function
"""
function target(t)
    return -sin(t)
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
    return sum([ (target(t)-nn(t, w, p))^2 for t in range ]) / length(range)
end

# Optimization
range = -2π:0.01:2π
nhl = 10; nw = 3*nhl+1; w0 = 20.0*rand(nw).-10.0; p = [nhl]
f = OptimizationFunction(loss, GalacticOptim.AutoZygote())
prob = OptimizationProblem(f, w0, p)
sol = solve(prob, ADAM(0.1), maxiters=8000)
prob = remake(prob,u0=sol.minimizer)
sol = solve(prob, ADAM(0.001), maxiters=2000)

# Plot result
plot([ target(t) for t in range ])
plot!([ nn(t, sol, p) for t in range ])






