### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ a904c8dc-c7fa-4ef7-97f0-9018bb9f2db2
begin
import Pkg
Pkg.add("Unitful")
Pkg.add("PeriodicTable")
Pkg.add("StaticArrays")
Pkg.add("LinearAlgebra")
Pkg.add("AtomsBase")
Pkg.add("InteratomicPotentials")
#Pkg.add(url="https://github.com/JuliaMolSim/AtomsBase.jl", rev="v0.1.0")
#Pkg.add(url="https://github.com/cesmix-mit/InteratomicPotentials.jl", rev="v0.1.2")
end

# ╔═╡ 5f6d4c16-bf9f-442a-bb43-5cbe1b861fb1
begin
using StaticArrays
using LinearAlgebra
using Unitful
using PeriodicTable
using AtomsBase
using InteratomicPotentials
using Flux
using Flux.Data: DataLoader
using CUDA
using PlutoUI
end

# ╔═╡ 5dd8b98b-967c-46aa-9932-25157a10d0c2
md" # Fitting atomic forces with a neural network using Julia"

# ╔═╡ 09985247-c963-4652-9715-1f437a07ef81
md" This notebook presents a step-by-step example about how to :

1) Generate simple surrogate DFT data
2) Define a neural network model
3) Train the model with the DFT data

Let's do it!
"

# ╔═╡ 3bdb681d-f9dc-4d37-8667-83fccc247b3d
md"## Installing and using the required packages"

# ╔═╡ e0e8d440-1df0-4581-8277-d3d6886351d7
md" What are these packages needed for?

- StaticArrays is used as the the type of small vectors such as forces. The speed of small SVectors, SMatrixs and SArrays is often > 10 × faster than Base.Array.
- LinearAlgebra is used to compute operations such as norms.
- Unitful is used to associate physical units to the parameters and variables.
- PeriodicTable is used to access information about the elements of the periodic table. In this notebook Argon is used.
- AtomsBase is used as an abstract interface for representation of atomic geometries. 
- InteratomicPotentials is used to computes methods (energies, forces, and virial tensors) for a variety of interatomic potentials, including the SNAP Potential (Thompson et al. 2014).
- Flux is used to define and train the neural network.
- CUDA is used to parallelize the training and execution of the neural network.
"

# ╔═╡ b562c28b-a530-41f8-b2d4-57e2df2860c5
md" ## Generating a simple surrogate DFT dataset"

# ╔═╡ 7bcb0360-a656-4637-b0ce-e4ada1e9ce0a
md"The following function generates a vector of atomic configurations. Each atomic configuration is defined based on [AtomsBase.jl](https://github.com/JuliaMolSim/AtomsBase.jl). In particular, a `FlexibleSystem` is used which contains information about the domain and its boundaries, as well as the atoms that compose it. The domain has zero Dirichlet boundary conditions, i.e. the probability that the atom reaches a boundary is zero. This example defines a binary system of Argon for each configuration. Each atom is represented by a `StaticAtom`. The positions of the atoms are calculated randomly within the domain under the constraint that both atoms are at a fixed distance. Modify this function if you want to change the distribution of the atoms :)"

# ╔═╡ 3e606585-86b6-4389-818d-bcbdb6078608
function gen_atomic_confs()
    # Domain
    L = 5u"Å"; σ0 = 2.0u"Å"
    box = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * L
    # Boundary conditions
    bcs = [DirichletZero(), DirichletZero(), DirichletZero()]
    # No. of atoms per configuration
    N = 2
    # No. of configurations
    M = 10000
    # Element
    elem = elements[:Ar]
    # Define atomic configurations
    atomic_confs = []
    for j in 1:M
        #σ0 = ((2.0 - 0.1) / (M - 1) * (j - 1) + 0.1)u"Å"
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

# ╔═╡ 19dc1efa-27ac-4c94-8f4d-bc7886c2959e
md"To fit the data a loss function will be defined. This function requires to precompute information about the neighbors of each atom. In particular, the position difference ($r_{i,j} = r_i - r_j$) between the atom $i$ to each of its neighbors ($j$) is precomputed."

# ╔═╡ c0d477c3-3af8-4b19-bf84-657d4c60fea8
md"$neighbors\_dists(i) = \{ r_{i,j} \ where \ j \  \epsilon \ neighbors(i)  \}$"

# ╔═╡ 4430121b-9d15-4c12-86f8-48fa1579845b
md"The neighbors $j$ of the atom $i$ are those within a radius $r_{cut}$ around it:"

# ╔═╡ 160f3fb3-8dbd-4a3b-befe-1bef361fdd69
md"$neighbors(i) = \{ j \neq i \  \backslash \ |r_i - r_j| < r_{cut} \}$"

# ╔═╡ 67d9544a-4ff4-4703-8579-b8a335d58f63
function compute_neighbor_dists(atomic_conf, rcutoff)
    neighbor_dists = []
    N = length(atomic_conf)
    for i in 1:N
        neighbor_dists_i = []
        for j in 1:N
            r = position(getindex(atomic_conf, i)) -
                position(getindex(atomic_conf, j))
            #r = SVector(map((x -> x.val), r.data)...)
			r = [map((x -> x.val), r.data)...]
            if 0 < norm(r) < rcutoff
                push!(neighbor_dists_i, r)
            end
        end
        push!(neighbor_dists, neighbor_dists_i)
    end
    return neighbor_dists
end

# ╔═╡ 0fa4ca5a-f732-4630-991c-ff5e4b76c536
md"Calculate the surrogate dft force ($f^{dft}_i$) of each atom ($i$) using the position difference ($r_{i,j} = r_i - r_j$) to each of its neighbors ($j$) and the potential $p$ (e.g. [LennardJones](https://en.wikipedia.org/wiki/Lennard-Jones_potential))."

# ╔═╡ 40c9d9cd-05af-4bbf-a772-5c09c1b03a66
md"$f^{dft}_i = \sum_{\substack{j \neq i \\ |r_i - r_j| < r_{cut}}} f^{dft}_{i,j}$"

# ╔═╡ 93450770-74c0-42f7-baca-7f8276373f9f
md"$f^{dft}_{i,j} = - \nabla LJ_{(r_{i,j})} = 24 ϵ  \left( 2 \left( \frac{σ}{|r_{i,j}|} \right) ^{12} -  \left( \frac{σ}{|r_{i,j}|} \right) ^6  \right) \frac{r_{i,j} }{|r_{i,j}|^2 }$"

# ╔═╡ 216f84d3-ec5c-4c49-a9d7-af0e98907e15
md"Here, the force between $i$ and $j$ is computed by the [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl) library."

# ╔═╡ 6e20dcb0-2fed-4637-ae07-775f3cd4a8dd
function compute_forces(neighbor_dists, p)
    return [ length(d)>0 ? sum(force.(d, [p])) : SVector(0.0, 0.0, 0.0)                            for d in neighbor_dists ]
end

# ╔═╡ 8aa8f56a-24a1-4ddd-840e-91517fd27b9c
md"The following function generates the training and test data sets consisting of the position differences with the neighbors of each atom and the surrogate DFT forces. The data sets are divided into batches using `Flux.DataLoader` and transferred to the GPU if necessary.devided in batches using `Flux.DataLoader` and transfer to the GPU if necessary."


# ╔═╡ 4e73e519-879b-447c-b125-387a5d1d7cd9
function gen_data(train_prop, batchsize, rcutoff, p, use_cuda, device)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Generate atomic configurations
    atomic_confs = gen_atomic_confs()

    # Generate neighbor distances (vectors)
    neighbor_dists = vcat(compute_neighbor_dists.(atomic_confs, rcutoff)...)
    N = floor(Int, train_prop * length(neighbor_dists))
    train_neighbor_dists = neighbor_dists[1:N]
    test_neighbor_dists = neighbor_dists[N+1:end]

    # Generate learning data using the potential `p` and the atomic configurations
    f_train = compute_forces(train_neighbor_dists, p)
    f_test = compute_forces(test_neighbor_dists, p)
    
    # If CUDA is used, convert SVector to Vector and transfer vectors to GPU
    if use_cuda
        train_neighbor_dists = device([device.(convert.(Vector, d)) for d in train_neighbor_dists])
        test_neighbor_dists =  device([device.(convert.(Vector, d)) for d in test_neighbor_dists])
        f_train = device.(convert.(Vector, f_train))
        f_test = device.(convert.(Vector, f_test))
    end

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((train_neighbor_dists, f_train), batchsize=batchsize, shuffle=true)
    test_loader  = DataLoader((test_neighbor_dists, f_test), batchsize=batchsize)

    return train_loader, test_loader
end

# ╔═╡ 152f05fb-ac0a-4d8d-8975-94585443853a
md" ## Defining the loss functions"

# ╔═╡ fe7cf55f-cbc9-41db-8554-03fdecbc881d
md"A simple loss function is defined. It calculates the root mean square error (rmse) between the surrogate DFT forces and the forces computed using the neural network model (defined below). The arguments of this loss function are batches of the training or test data set, N is the batch size."

# ╔═╡ dd5e6c1f-ad9b-49fa-8b5e-9cffd5781c16
md"$loss(f^{model}, f^{dft}) = \sqrt{\frac{ {\sum_{i} |f^{model}_i - f^{dft}_i| ^2} } {N}}$"

# ╔═╡ 7b48d757-18eb-4f6b-bf7f-53d8c52cf7a1
loss(f_model, forces) = sqrt(sum(norm.(f_model .- forces).^2) / length(forces))

# ╔═╡ f93f298d-6ac0-45e2-abee-ab36d85b3f7b
md"The force of the atom $i$ is computed as"

# ╔═╡ 1180e03c-482e-428a-b66c-5534bcb4de83
md"$f^{model}_i = \sum_{r_{i,j} \ \epsilon \ neighbors\_dists(i) } model(r_{i,j})$"

# ╔═╡ 08fbcf33-0935-41ce-a721-c597e2570b47
md"The following function computes $f^{model}$ for a batch of atoms"

# ╔═╡ 8a03ffbc-9ab2-438d-b71b-1d7238b33507
f_model(d, model) = [length(di)>0 ? sum(model.(di)) : zeros(3) for di in d]

# ╔═╡ 2c629ed0-e21d-4f38-b46f-fc2a1757b62f
md"The global loss or loss of an entire data set (training or test) is the average of the losses of the batches in that data set."

# ╔═╡ 48e7fa16-ea5f-44f5-84ab-2994e7382c4e
md"$\sum_{k \ \epsilon \ \#batches(training\_set)} loss(f^{model}_k, f^{dft}_k)$"

# ╔═╡ 7410e309-2ee0-4b65-af2a-c4f738a3bd80
global_loss(loader, model) = 
        sum([loss(f_model(d, model), f) for (d, f) in loader]) / length(loader)

# ╔═╡ 2244cd54-f927-40e0-8fad-75a34096044d
md"More complex examples can also fit energies and tensors, and/or define hybrid potentials."

# ╔═╡ b481e2f6-1a87-4af3-8d93-111e2ab8d933
md"## Defining and training the model in CPU"

# ╔═╡ 16a90976-77e9-4ea0-99bd-8278d9812681
md"The first step is to call the functions defined above to generate the training and test data sets. To do this, the Lennard-Jones interatomic potential is defined using [InteratomicPotentials.jl](https://github.com/cesmix-mit/InteratomicPotentials.jl)"

# ╔═╡ 93e7f0c4-ad8c-473e-a8d3-09508f2ba6df
begin
lj_ϵ = 1.0
lj_σ = 1.0
lj = LennardJones(lj_ϵ, lj_σ);
end

# ╔═╡ f1a29169-2c8a-41bc-9c35-be2c77b418ec
md"To finally obtain the data sets, the interatomic potential is used in combination with other input parameters: the radius of the neighborhoods ($rcutoff$), the training size ratio ($train\_prop$), the batch size ($batchsize$), and parameters related to CPU and GPU usage ($use\_cuda$ and $device1$)."

# ╔═╡ 54891249-99be-4798-a0e9-5b0193ecaa3c
begin
rcutoff = 2.5 * lj_σ
train_prop = 0.8
batchsize = 256
use_cuda = false
device1 = cpu
cpu_train_loader, cpu_test_loader = gen_data(train_prop, batchsize, rcutoff, lj, use_cuda, device1)
end

# ╔═╡ 1cb8b6d0-d645-4d16-a501-cc9d2a047b39
md"The neural network model and the parameters to be optimized using [Flux.jl](https://github.com/FluxML/Flux.jl). An explanation about `Chain` and `Dense` is presented [here](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain)."

# ╔═╡ c8c43e19-daff-4272-8703-a2dcaceca7a9
begin
cpu_model = Chain(Dense(3,150,Flux.σ),Dense(150,3))
cpu_ps = Flux.params(cpu_model) # model's trainable parameters
end

# ╔═╡ fcccc8ca-97c8-43cd-ab1d-b76f126ab617
md"The optimizer [ADAM](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) is defined using a learning rate η = 0.1."

# ╔═╡ 49d7f00a-db88-4dab-9209-c9840255d999
opt = ADAM(0.1)

# ╔═╡ 4481fcae-6c0c-4455-9a11-0eb1f19805cc
md"The model is trained in the loop below."

# ╔═╡ 2e199532-5177-4b39-9c7e-83d61c1e2f13
begin
epochs = 20
with_terminal() do
	for epoch in 1:epochs
	    # Training of one epoch
	    time = Base.@elapsed for (d, f) in cpu_train_loader
	        gs = gradient(() -> loss(f_model(d, cpu_model), f), cpu_ps)
	        Flux.Optimise.update!(opt, cpu_ps, gs)
	    end
	    
	    # Report traning loss
	    println("Epoch: $(epoch), loss: $(global_loss(cpu_train_loader, cpu_model)), time: $(time)")
	end
end
end

# ╔═╡ 74d94c45-7a66-4779-9bd8-d6987d23bdc4
md"### Test CPU results"

# ╔═╡ 46a7f1ab-5f99-4b69-9bff-c299815cccab
md"Root mean squared error (rmse)"

# ╔═╡ 195925f3-19ea-47a2-bc31-561bf698f580
@show "Test RMSE: $(global_loss(cpu_test_loader, cpu_model))"

# ╔═╡ 4d94701b-e3a5-4aef-ba23-97c1150f89a1
md"## Defining and training the model in GPU"

# ╔═╡ c2492af2-23e7-45c5-bf44-109c59d7986d
md"This case is analogous to that of the CPU except for some differences. First, when training and test data are generated, they must be transferred to the GPU. This is done in the `gen_data` function."

# ╔═╡ d4526269-07e6-4684-8bed-7cee5898ef52
begin
use_cuda_ = true
device2 = gpu
gpu_train_loader, gpu_test_loader = gen_data(train_prop, batchsize, rcutoff, lj, use_cuda_, device2)
end

# ╔═╡ 5a7e8a3f-3d8d-4048-80fe-c64e9c1cfd44
begin
gpu_model = Chain(Dense(3,150,Flux.σ),Dense(150,3)) |> device2
gpu_ps = Flux.params(gpu_model) # model's trainable parameters
end

# ╔═╡ f89622bb-433a-4547-aec0-9fd3a2ffbaf2
md"The model is trained in the loop below. In this case the elapsed time is measured using `CUDA.@elapsed`."

# ╔═╡ c6e66ff5-b5f2-4c0f-9eae-f123034bd438
begin
epochs_ = 20
with_terminal() do
	for epoch in 1:epochs_
	    # Training of one epoch
	    time = CUDA.@elapsed for (d, f) in gpu_train_loader # or 
	        gs = gradient(() -> loss(f_model(d, gpu_model), f), gpu_ps)
	        Flux.Optimise.update!(opt, gpu_ps, gs)
	    end
	    
	    # Report traning loss
	    println("Epoch: $(epoch), loss: $(global_loss(gpu_train_loader, gpu_model)), time: $(time)")
	end
end
end

# ╔═╡ 24f677cf-ee87-4af0-9c11-5d0911f1c700
md"### Test GPU results"

# ╔═╡ d300f5ce-2cce-4739-9108-ef20ff889955
@show "Test RMSE: $(global_loss(gpu_test_loader, gpu_model))"

# ╔═╡ Cell order:
# ╟─5dd8b98b-967c-46aa-9932-25157a10d0c2
# ╟─09985247-c963-4652-9715-1f437a07ef81
# ╟─3bdb681d-f9dc-4d37-8667-83fccc247b3d
# ╟─e0e8d440-1df0-4581-8277-d3d6886351d7
# ╠═a904c8dc-c7fa-4ef7-97f0-9018bb9f2db2
# ╠═5f6d4c16-bf9f-442a-bb43-5cbe1b861fb1
# ╟─b562c28b-a530-41f8-b2d4-57e2df2860c5
# ╟─7bcb0360-a656-4637-b0ce-e4ada1e9ce0a
# ╠═3e606585-86b6-4389-818d-bcbdb6078608
# ╟─19dc1efa-27ac-4c94-8f4d-bc7886c2959e
# ╟─c0d477c3-3af8-4b19-bf84-657d4c60fea8
# ╟─4430121b-9d15-4c12-86f8-48fa1579845b
# ╟─160f3fb3-8dbd-4a3b-befe-1bef361fdd69
# ╠═67d9544a-4ff4-4703-8579-b8a335d58f63
# ╟─0fa4ca5a-f732-4630-991c-ff5e4b76c536
# ╟─40c9d9cd-05af-4bbf-a772-5c09c1b03a66
# ╟─93450770-74c0-42f7-baca-7f8276373f9f
# ╟─216f84d3-ec5c-4c49-a9d7-af0e98907e15
# ╠═6e20dcb0-2fed-4637-ae07-775f3cd4a8dd
# ╟─8aa8f56a-24a1-4ddd-840e-91517fd27b9c
# ╠═4e73e519-879b-447c-b125-387a5d1d7cd9
# ╟─152f05fb-ac0a-4d8d-8975-94585443853a
# ╟─fe7cf55f-cbc9-41db-8554-03fdecbc881d
# ╟─dd5e6c1f-ad9b-49fa-8b5e-9cffd5781c16
# ╠═7b48d757-18eb-4f6b-bf7f-53d8c52cf7a1
# ╟─f93f298d-6ac0-45e2-abee-ab36d85b3f7b
# ╟─1180e03c-482e-428a-b66c-5534bcb4de83
# ╟─08fbcf33-0935-41ce-a721-c597e2570b47
# ╠═8a03ffbc-9ab2-438d-b71b-1d7238b33507
# ╟─2c629ed0-e21d-4f38-b46f-fc2a1757b62f
# ╟─48e7fa16-ea5f-44f5-84ab-2994e7382c4e
# ╠═7410e309-2ee0-4b65-af2a-c4f738a3bd80
# ╟─2244cd54-f927-40e0-8fad-75a34096044d
# ╟─b481e2f6-1a87-4af3-8d93-111e2ab8d933
# ╟─16a90976-77e9-4ea0-99bd-8278d9812681
# ╠═93e7f0c4-ad8c-473e-a8d3-09508f2ba6df
# ╟─f1a29169-2c8a-41bc-9c35-be2c77b418ec
# ╠═54891249-99be-4798-a0e9-5b0193ecaa3c
# ╟─1cb8b6d0-d645-4d16-a501-cc9d2a047b39
# ╠═c8c43e19-daff-4272-8703-a2dcaceca7a9
# ╟─fcccc8ca-97c8-43cd-ab1d-b76f126ab617
# ╠═49d7f00a-db88-4dab-9209-c9840255d999
# ╟─4481fcae-6c0c-4455-9a11-0eb1f19805cc
# ╠═2e199532-5177-4b39-9c7e-83d61c1e2f13
# ╟─74d94c45-7a66-4779-9bd8-d6987d23bdc4
# ╟─46a7f1ab-5f99-4b69-9bff-c299815cccab
# ╠═195925f3-19ea-47a2-bc31-561bf698f580
# ╟─4d94701b-e3a5-4aef-ba23-97c1150f89a1
# ╟─c2492af2-23e7-45c5-bf44-109c59d7986d
# ╠═d4526269-07e6-4684-8bed-7cee5898ef52
# ╠═5a7e8a3f-3d8d-4048-80fe-c64e9c1cfd44
# ╠═f89622bb-433a-4547-aec0-9fd3a2ffbaf2
# ╠═c6e66ff5-b5f2-4c0f-9eae-f123034bd438
# ╟─24f677cf-ee87-4af0-9c11-5d0911f1c700
# ╠═d300f5ce-2cce-4739-9108-ef20ff889955
