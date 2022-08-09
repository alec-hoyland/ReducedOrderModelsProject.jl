# Reduced Order Models for Solving Partial Differential Equations
A personal project investigating and benchmarking reduced order models for partial differential equations.

In this project, I derive a formulation for a reduced order model that exploits the structure of a partial differential equation to solve it at high accuracy at arbitrary parameter values, using far fewer FLOPs to do so than with traditional linear system solvers, numerical simulation, or approximation by neural networks (surrogates).

**For more information, see [here](https://alec-hoyland.github.io/ReducedOrderModelsProject.jl/).**

### What is a reduced order model?

A reduced order model (ROM) is a smaller model that mimics the results of a larger model. Often in mathematical modeling and numerical simulation, you run into issues with increasing computational complexity where the state spaces gets too large, or numerical simulations become infeasible. ROMs alleviate this problem by providing a smaller model that can be used in simulations or numerical computations instead. A good ROM has a small approximation error compared to the full order model and conserves the properties of the larger model.

### How does this work?

We start with a one-dimensional diffusion partial differential equation] with known initial and boundary conditions and take the Laplace transform to get an easier system to solve. This system has one parameter and our goal is to find a way to get the solution of the system at any arbitrary parameter value.

We discretize the PDE to form a linear system. Given the solution at a few parameter values, we can start to build our data-driven ROM.

![](https://raw.githubusercontent.com/alec-hoyland/ReducedOrderModelsProject.jl/e56096a2a190eb42e43955e4d4727b9b81b2a8e4/latex/Images/fig1.svg)

The key idea here is that we can construct a rational Krylov subspace where the vectors in the subspace are solutions at different parameter values. A solution for an arbitary parameter value is a linear combination of those solutions from the subspace. Thus, the Krylov subspace forms a basis for the solutions of the PDE system.

It turns out that we don’t need to know all the vectors in the subspace; we only need about 10. The ROM is constructed by looking at the differences between solutions and accurately captures the dynamics with only a few examples. This is far better performance than a neural network function approximator because vanilla NNets don’t exploit the known dynamics for the system. Here, we can write the equation of state, so we can take advantage of it directly.

