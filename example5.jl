# Using Flux and Zygote to fit a model composed by two sub-models

using Flux, Plots

# Data range
r = -2π:0.01:2π

# Target function
f(x) = sin(x)

# Predicted function

# Define two simple neural network models
model1 = Chain(
    Dense(1, 16, sigmoid),
    Dense(16, 16, sigmoid),
    Dense(16, 1)
)
model2 = Chain(
    Dense(1, 16, sigmoid),
    Dense(16, 16, sigmoid),
    Dense(16, 1)
)
models = [model1, model2]

f_pred(models, x) = sum([m([x])[1] for m in models])

# Synthetic data
data = [(Float32(x), Float32(sin(x))) for x in r]

# Loss function that uses an array of models
function loss(models, data)
    loss = 0.0
    for (x, y) in data
        loss += (f_pred(models, x) - y[1])^2
    end
    return loss / length(data)
end

# Training
∇loss(models, data) = Flux.gradient((models) -> loss(models, data), models)
optim = Flux.setup(Adam(1e-2), models)
n_epochs = 2000
for epoch in 1:n_epochs
    grads = ∇loss(models, data)
    Flux.update!(optim, models, grads[1])
    if epoch % 100 == 0
        l = loss(models, data)
        println(l)
    end
end

# Plot results
plot(r, f.(r))
plot!(r, f_pred.([models], r))

