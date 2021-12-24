"""

Fitting atomic forces using a neural network

"""

using Unitful, PeriodicTable, StaticArrays, LinearAlgebra
using AtomsBase
#using ElectronicStructure
using InteratomicPotentials
#using PotentialLearning

using Flux, Statistics
using Flux.Data: DataLoader
using Flux: @epochs
using Base: @kwdef
using CUDA

################################################################################
# Data generation functions
################################################################################

"""
    gen_test_atomic_conf()

Generate test atomic configurations.
Each configuration has two argon atoms whose positions are random but have
a fixed distance between them.
"""
function gen_test_atomic_conf()
    # Domain
    L = 5u"Å"; σ0 = 2.0u"Å"
    box = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * L
    # Boundary conditions
    bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
    # No. of atoms per configuration
    N = 2
    # No. of configurations
    M = 100
    # Element
    elem = elements[:Ar]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        atoms = []
        ϕ = rand() * 2.0 * π; θ = rand() * π
        x = (L/2.0-σ0) * cos(ϕ) * sin(θ) + L/2.0
        y = (L/2.0-σ0) * sin(ϕ) * sin(θ) + L/2.0
        z = (L/2.0-σ0) * cos(θ) + L/2.0
        pos1 = SVector{3}(x, y, z)
        atom = StaticAtom(pos1, elem)
        push!(atoms, atom)
        ϕ = rand() * 2.0 * π; θ = rand() * π
        x += σ0 * cos(ϕ) * sin(θ)
        y += σ0 * sin(ϕ) * sin(θ)
        z += σ0 * cos(θ)
        pos2 = SVector{3}(x, y, z)
        atom = StaticAtom(pos2, elem)
        push!(atoms, atom)
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

"""
    gen_test_atomic_conf_2()

Generate test atomic configurations. Atoms positions are random.
"""
function gen_test_atomic_conf_2()
    # Domain
    L = 5u"Å"
    box = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * L
    # Boundary conditions
    bcs = [Periodic(), Periodic(), DirichletZero()]
    # No. of atoms per configuration
    N = 2
    # No. of configurations
    M = 80
    # Element
    c = elements[:C]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        atoms = []
        for i in 1:N
            pos = SVector{3}(rand(3)*L...)
            atom = StaticAtom(pos,c)
            push!(atoms, atom)
        end
        push!(atomic_confs, FlexibleSystem(box, bcs, atoms))
    end
    return atomic_confs
end

"""
    compute_neighbor_dists(atomic_conf, rcutoff)

Calculate for each atom the distances (vector) to each of its neighbors.
"""
function compute_neighbor_dists(atomic_conf, rcutoff)
    neighbor_dists = []
    N = length(atomic_conf)
    for i in 1:N
        neighbor_dists_i = []
        for j in 1:N
            r = position(getindex(atomic_conf, i)) -
                position(getindex(atomic_conf, j))
            r = SVector(map((x -> x.val), r.data))
            if 0 < norm(r) < rcutoff
                push!(neighbor_dists_i, r)
            end
        end
        push!(neighbor_dists, neighbor_dists_i)
    end
    return neighbor_dists
end

"""
    compute_forces(neighbor_dists, p)

Calculate the force of each atom using the distance (vector) to each of its
neighbors and the potential `p`.
"""
function compute_forces(neighbor_dists, p)
    f = []
    for d in neighbor_dists
        if length(d)>0
            aux = sum(force.(d, [p]))
        else
            aux = [0.0, 0.0, 0.0]
        end
        push!(f, Float32.(aux))
    end
    return f
end

"""
    gen_data(train_prop, batchsize, p, rcutoff)

Compute neighbor distances and forces for training and validation.
"""
function gen_data(train_prop, batchsize, rcutoff, p)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Generate test atomic configurations: domain and particles (position, velocity, etc)
    atomic_confs = gen_test_atomic_conf()

    # Generate neighbor distances (vectors)
    neighbor_dists = vcat(compute_neighbor_dists.(atomic_confs, rcutoff)...)
    N = floor(Int, train_prop * length(neighbor_dists))
    train_neighbor_dists = neighbor_dists[1:N]
    test_neighbor_dists = neighbor_dists[N+1:end]

    # Generate learning data using the potential `p` and the atomic configurations
    f_train = compute_forces(train_neighbor_dists, p)
    f_test = compute_forces(test_neighbor_dists, p)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((train_neighbor_dists, f_train), batchsize=batchsize)
    test_loader  = DataLoader((test_neighbor_dists, f_test), batchsize=batchsize)

    return train_loader, test_loader
end

################################################################################
# Model definition
################################################################################

# GPU usage
use_cuda = false  # use gpu (if cuda available)
if CUDA.functional() && use_cuda
    @info "Training on CUDA GPU"
    CUDA.allowscalar(true)
    device1 = gpu
else
    @info "Training on CPU"
    device1 = cpu
end

# Input data: create test and train dataloaders
train_prop = 0.8; batchsize = 40
lj_ϵ = 1.0; lj_σ = 1.0; rcutoff = 2.5*lj_σ; lj = LennardJones(lj_ϵ, lj_σ);
train_loader, test_loader = gen_data(train_prop, batchsize, rcutoff, lj)

# Model: neural network
model = Chain(Dense(3,20,Flux.σ),Dense(20,3)) |> device1
ps = Flux.params(model) # model's trainable parameters

# Optimizer
η = 3e-4 # learning rate
opt = ADAM(η)

# Loss function
loss(model, neighbor_dists, forces) = 
            sqrt(sum( [ length(d)>0f0 ? norm(sum(model.(d)) - f)^2 : 0f0
                        for (d, f) in zip(neighbor_dists, forces)]))

################################################################################
# Training
################################################################################

epochs = 1000
for epoch in 1:epochs
    # Training of one epoch
    for (d, f) in train_loader
        d, f = device1(d), device1(f) # transfer data to device
        gs = gradient(() -> loss(model, d, f), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end

    # Report traning loss
    train_loss_sum = 0.0
    for (d, f) in train_loader
        d, f = device1(d), device1(f)
        train_loss_sum += loss(model, d, f)
    end
    println("Epoch:", epoch, ", loss:", train_loss_sum / length(train_loader))

end
println("")


################################################################################
# Validation
################################################################################

max_rel_error = 0.0; max_abs_error = 0.0
for (d, f) in test_loader
    d, f = device1(d), device1(f) # transfer data to device
    
    aux = maximum([ length(d0)>0 ? norm(sum(model.(d0)) - f0) / norm(sum(model.(d0))) : 0.0
                    for (d0, f0) in zip(d, f)])
    if aux > max_rel_error
        global max_rel_error = aux
    end
    
    aux = maximum( [ length(d0)>0 ? norm(sum(model.(d0)) - f0) : 0.0
                     for (d0, f0) in zip(d, f)])
    if aux > max_abs_error
        global max_abs_error = aux
    end
end
println("Maximum relative error: ", max_rel_error)
println("Maximum absolute error: ", max_abs_error)


