### A Pluto.jl notebook ###
# v0.19.2

using Markdown
using InteractiveUtils

# ╔═╡ dd578ceb-8038-4967-ac01-6fb337b3328b
using LinearAlgebra, Plots, Random, Interpolations

# ╔═╡ fc4be02c-c27c-11ec-1c65-d5400975275e
md"# Data-Driven Reduced-Order Models for Physics-Based Machine Learning"

# ╔═╡ 4a9bfc40-d165-4fa5-8b36-e1db08d98249
md"""

## Introduction

In this notebook, we derive a formulation for a reduced-order model that exploits the structure of a partial differential equation, to solve it with high accuracy at arbitrary parameter values, using far fewer FLOPs to do so than by traditional linear system solvers, numerical simulation, or neural networks."""

# ╔═╡ 5b7b9d6f-f244-41d5-b4c9-675470aa7df7
md"""
Consider the one-dimensional diffusion partial differential equation:

$$- \frac{\partial}{\partial x} \bigg(k(x) \frac{\partial u}{\partial x}\bigg) + \frac{\partial u}{\partial x} = 0$$

for $x \in [0, 1]$ with initial conditions $u(t = 0) = \delta (x - x_0)$ and boundary conditions $u(0) = u(1) = 0$. In the Laplace frequency-domain, the equation takes the form: 

$$-\frac{\partial}{\partial x} \bigg( k \frac{\partial \hat{u}}{\partial x} \bigg) + \lambda \hat{u} = \delta (x - x_0)$$

For $\lambda \in \{\lambda_1, ..., \lambda_k\}$, let our data $d$, be the solution to the above equation evaluated at $\tilde{x}$ such that $d(\lambda) = \hat{u}(\lambda, \tilde{x})$ and $d'(\lambda) = \hat{u}'(\lambda, \tilde{x})$, where $\tilde{x}$ is some point on the the support of $x$.
"""

# ╔═╡ 9cfae88e-f6f3-4364-816e-8aad0f1a1033
md"""
Let $x_i = ih$ for $i = 0, ..., n$ and $h = \frac{1}{n}$, be a uniform grid on the segment $[0, 1]$.

We discretize our equation for $\tilde{x} = x_1$ on that grid:

$$Au + \lambda u = \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} + \lambda u_i = e_1$$

for $i = 1, ..., n-1$, such that $u_i \approx u(x_i)$ is a discrete approximation of $u(x)$. We also assume that our measurements are approximated by $e_1^{*}u$.
"""

# ╔═╡ b2807c3a-bd5a-443f-afde-0fc8a340540a
md"## Construction of the data-driven ROM"

# ╔═╡ 0613a3f2-4d5d-4b3b-87a4-f9695f2ec43c
md"""
We will treat $u_i = u_i(x) = u(x, \lambda_i)$ as a continuous basis function (or equivalent, an infinitely dimensional column vector). The projection subspace is then:

$$\mathbb{U} = \mathrm{span}\big\{ u_1(x), u_2(x), ..., u_{n-1}(x) \big\}$$

The data-driven ROM is the Galerkin system for this subspace:

$$S c(\lambda) + \lambda M c(\lambda) = b$$

where $b \in \mathbb{C}^{n-1}$ is a column vector with components $b_j = \langle u_j, e_1 \rangle$ and $c(\lambda) \in \mathbb{C}^{n-1}$ is a column vector of coefficients of the solution with respect to the basis of exact solutions $\mathbb{U}$.

Using the condition that the transfer function $\tilde{F}(\lambda) = b^*c$, we obtain two relationships, by taking conjugate inner products of the differential equation with $\bar{u}_j$ or $u_i$. First that

$$\bar{S}_{ij} + \lambda_i \bar{M}_{ij} = \bar{F}(\lambda_j)$$

and secondly that

$$\bar{S}_{ij} + \lambda_j M_{ij} = \bar{F}(\lambda_i)$$

Therefore

$$M_{ij} = \frac{\bar{F}(\lambda_i) - F(\lambda_j)}{\lambda_j - \lambda_i}$$

and

$$S_{ij} = \frac{F(\lambda_j)\lambda_j - \bar{F}(\lambda_i)\lambda_i}{\lambda_j - \bar{\lambda}_i}$$
"""

# ╔═╡ def01aeb-7f2c-431f-b391-e9209d30008d
md"""
Using this formulation, we can say that the solution to the differential equation, $u(\lambda)$, is close to the Galerkin projection

$$u(\lambda) \approx \tilde{u}(\lambda) = V c(\lambda) = V(S + \lambda M)^{-1}b$$

where $V$ is a matrix of basis functions, $V = [u_1, ..., u_{n-1}]$.
"""

# ╔═╡ 4275508e-9180-48da-99d7-4a350b6f1d5a
md"### Relationship to rational Krylov subspaces"

# ╔═╡ 85d1c4af-d4ae-4f17-a4f1-f39cb0ad3f3a
md"""
Consider a rational Krylov subspace $\mathcal{V}_k = \mathrm{span}\big\{ (A + \lambda_1)^{-1} e_1, ..., (A + \lambda_k)^{-1} e_1 \big\}$.

Denote the matrix of concatenated columns $1,...,k$ as $V_k = [v_1, v_2, ..., v_k] =  \big[ (A + \lambda_1)^{-1}e_1, ..., (A + \lambda_k)^{-1}e_1 \big] \in \mathbb{R}^{(n-1)\times k}$. This matrix is the same as the matrix, $V$, of vectors that span $\mathbb{U}$, except that only $k$ columns are included.
"""

# ╔═╡ e87ac081-5a5b-4886-a8bd-0a0ef4c820e8
md"""
Let $v_i$ and $v_j$ be the $i$-th and $j$-th column vectors of $V_k$. By definition of the rational Krylov subspace:

$$A v_i + \lambda v_i = e_1$$

and 

$$A v_j + \lambda v_j = e_1$$

Left-multiplying these equalities by $v^*_j$ and $v^*_i$ respectively:

$$v_j^* A v_i + \lambda v_j^* v_i = v^*_j e_1$$

$$v^*_i A v_j + \lambda v_i^* v_j = v^*_i e_1$$

### Defining the $M$ and $S$ matrices

From the definitions of matrices $S$ and $M$ above, by exploiting the symmetry:

$$S_{ij} + \lambda M_{ij} = v^*_je_1$$

and 

$$S_{ij} + \lambda M_{ij} = v^*_ie_1$$

Since $v^*_j = \big((A + \lambda_j)^{-1} e_1 \big)^* = e^*_1 (A + \lambda_j)^{\dagger}$ and $(A + \lambda_j)^{-1}$ is symmetric tridiagonal, $(A + \lambda_j)^{-1} = (A + \lambda_j)^{\dagger}$, then $v^*_j = e^*_j (A + \lambda_j)^{-1}$.

We note that $v^*_je_1=e_j^*(A + \lambda_j)^{-1}e_1 = e_1^* u = d(\lambda_j)$, which we defined earlier. This is true for any $j \in 1, ..., k$.
"""

# ╔═╡ 65646978-f09c-47d1-a60e-3b3aca8b54aa
md"""
### Computing $d^\prime(\lambda)$
To find the values of the $S$ and $M$ matrices along the diagonal, *i.e.*, $S_{ii}$ and $M_{ii}$, we consider the limit $i \rightarrow j$. Given the definition of the limit:

$$\lim_{\lambda_i \rightarrow \lambda_j} \frac{f(\lambda_j) - f(\lambda_i)}{\lambda_i - \lambda_j}$$

It is clear to see that 

$$S_{ii} = \lim_{\lambda_i \rightarrow \lambda_j} \frac{\lambda_i d(\lambda_i) - \lambda_j d(\lambda_j)}{\lambda_i - \lambda_j} = \big(\lambda d(\lambda)\big)^\prime|_{\lambda=\lambda_i}$$

and similarly, that $M_{ii} = -d^\prime(\lambda_i)$.
"""

# ╔═╡ cb698a86-4499-4189-85cd-326711efffa3
md"""
By definition: $d^\prime (\lambda) = \hat{u}^\prime(\lambda, x^{\prime\prime})$. Since

$$Au + \lambda u = e_1$$

we can compute

$$\frac{\partial}{\partial \lambda}\big(Au + \lambda u \big) = \frac{\partial}{\partial \lambda} e_1 = \vec{0} = (A + \lambda\mathbb{I})\frac{\partial u}{\partial \lambda} + u$$

$$\frac{\partial u}{\partial \lambda} = - (A + \lambda\mathbb{I})^{-1}u$$

By using the property that our measurements are approximated by $e_1^* u$:

$$d^\prime(\lambda) = e^*_1 \frac{\partial u}{\partial \lambda}$$

Substituting and rearranging:

$$d^\prime(\lambda) = - e_1^*(A + \lambda\mathbb{I})^{-1}u =  (-(A + \lambda\mathbb{I})^{-1}e_1)^* u = -u^*u$$
"""

# ╔═╡ 8281f11f-3b6b-4b8c-91a9-274ba6d29690
md"""
### Solving the system

The Galerkin system is:

$$S w + \hat{\lambda} M w = \vec{d}$$

where $\vec{d} = [d(\lambda_1), d(\lambda_2), ..., d(\lambda_k)]$, and $S, M \in \mathbb{R}^{k \times k}$ and $w \in \mathbb{R}^k$ is a vector of coefficients for the solution in the Krylov basis. $\hat{\lambda}$ is any arbitrary $\lambda \notin [\lambda_1, ..., \lambda_k]$, for which we want a solution $u(\hat{\lambda},x^{\prime\prime})$.

We solve this linear system for $w$ and then receive the approximate solution 

$$u(\hat{\lambda},x^{\prime\prime}) = V_k w_k = \vec{d}^*w$$
"""

# ╔═╡ 46a87084-e158-4fa7-b335-946b2d76a689
md"""## Results"""

# ╔═╡ 0e4bbebb-87dc-46c6-afce-41a77d6cd821
md"We define the matrix $A \in \mathbb{R}^{n \times n}$ with $n=100$. It is symmetric and tridiagonal."

# ╔═╡ 1579cb7d-fd21-4ca1-8ca1-111e08ba654a
n = 100

# ╔═╡ 388c4deb-4be6-4dbe-b0fa-b1f8bb20b418
x = range(0, 1, n+1)

# ╔═╡ 538f2513-7ca0-493d-8269-9a53c55e4832
h = 1/(n+1)

# ╔═╡ 4f6f61f3-32cc-4aa0-8142-979cb55b0274
A = - (1/h^2) * SymTridiagonal(fill(-2.0, n), fill(1.0, n-1))

# ╔═╡ c68cea83-33a6-4cac-a88d-5e8f7d68c4ed
x′′ = x[2]

# ╔═╡ 4dce59ce-8d70-4588-b5a5-856524c79c88
begin
	e_1 = zeros(n)
	e_1[1] = 1
	e_1
end

# ╔═╡ 94ac5af7-2c5d-4d5b-ba9b-6a215a95401b
"""
	get_M(λs::T, ds::T, d′s::T) where T

Function to create the `M` matrix using `λ`, `d(λ)`, and `d′(λ)` values.
"""
function get_M(λs, ds, d′s)
	@assert length(ds) == length(d′s)
	n = length(ds)
	M = zeros(n, n)
	for i in 1:n, j in 1:n
		if i != j
			M[i, j] = (ds[j] - ds[i]) / (λs[i] - λs[j])
		end
	end
	for i in 1:n
		M[i, i] = - d′s[i]
	end

	return M
end

# ╔═╡ 57d63ae6-2032-43a3-8333-0216e0e6f45d
"""
	get_S(λs::T, ds::T, d′s::T) where T

Function to create the `S` matrix using `λ`, `d(λ)`, and `d′(λ)` values.
"""
function get_S(λs, ds, d′s)
	@assert length(λs) == length(ds) == length(d′s)
	n = length(λs)
	S = zeros( n, n)
	for i in 1:n, j in 1:n
		if i != j
			S[i, j] = (λs[i] * ds[i] - λs[j] * ds[j]) / (λs[i] - λs[j])
		end
	end
	for i in 1:n
		S[i, i] = λs[i] * d′s[i] + ds[i] # (λd(λ))′ evaluated at λ_i
	end

	return S
end

# ╔═╡ e2e40cdc-6806-4648-8c0e-562ec2238840
"""
	get_data(λ::T) where T <: Number

Function to explicitly solve the system for a given `λ` value. Returns `d(λ)` and `d′(λ)`.
"""
function get_data(λ::T) where T <: Number
	u = (A + λ*I) \ e_1 # (A + λ)^-1 * e_1
	u′ = -u'u # u′ evaluated at x′′
	return u[1], u′
end

# ╔═╡ 6662e106-5309-4d0a-ad65-c5f13b1e6705
md"We sample $\lambda \in [0, 500]$ along a log scale. We interleave values, to better sample the full range of parameters with increasing $k$."

# ╔═╡ 26f4949d-b2b7-4e3e-a311-f287210aa7b8
"""
	interleave(x)

Interleave an iterable `x`, so that the first element is followed by the last and so on, e.g. `x = [1, 2, 3, 4]` becomes `y = [1, 4, 2, 3]`.
"""
function interleave(x)
	n = length(x)
	if n % 2 == 0
		k::Int = n/2
		x_1 = x[1:k]
		x_2 = reverse(x[(k+1):end])
		return [x_1 x_2]'[:]
	else
		error("`length(x)` cannot be odd, not implemented")
		return nothing
	end		
end

# ╔═╡ 49437688-2398-4f41-92d7-b4ae50dc413e
λ_train = 10.0 .^ collect(range(log10(10), log10(500); length=40)) .|> round |> interleave

# ╔═╡ f77d569d-b209-4f2a-9ced-0ce144e26b2f
λ_test = rand(10:500, 11)

# ╔═╡ 0aa19f8b-be62-49e5-9942-61f798688e3c
λ_train_linear = range(10, 500, 40) |> collect |> interleave

# ╔═╡ c68c27ca-f3a6-408f-8d6b-d50c51bde1d1
md"Solve the system directly for $\lambda$ values in the training set."

# ╔═╡ cb08c667-fb53-4688-bbfe-c9c045a82a1b
output = [get_data(λ_k) for λ_k in λ_train]

# ╔═╡ b5d1385c-066b-4a06-9815-c75b5f4767ff
ds = [i[1] for i in output]

# ╔═╡ 9a9dd256-f669-4397-aff3-a7f37055e637
d′s = [i[2] for i in output]

# ╔═╡ bb4f009b-6b87-4c5a-a727-be36956a33a3
"""
Function to solve the Galerkin system using data `d` (with derivative `d′`) corresponding to parameter `λ`.
"""
function solve(λs, λ, ds, d′s)
	@assert length(λs) == length(ds) == length(d′s)
	M = get_M(λs, ds, d′s)
	S = get_S(λs, ds, d′s)
	w_k = (S + λ * M) \ ds
	return ds' * w_k
end

# ╔═╡ 495da75a-cf51-4424-ab48-54bead162ce8
"""
	get_error(x, x̂)

Function to compute the error via L2 norm.
"""
function get_error(x, x̂)
	norm(x - x̂)
end

# ╔═╡ 8d8fb253-1e4d-4786-86ff-0c195a61f0e8
"""
	do_experiment(λ_train, λ_test, ds, d′s)

Compute the error, averaged over all samples of `λ` in the testing set, for increasing values of `k`.
"""
function do_experiment(λ_train, λ_test, ds, d′s)
	N = length(λ_train)
	err = zeros(N)
	for k in 1:N
		for this_λ in λ_test
			try
				d̂ = solve(λ_train[1:k], this_λ, ds[1:k], d′s[1:k])
				err[k] += get_error(d̂, get_data(this_λ)[1])
			catch
				d̂ = NaN
				err[k] = NaN
			end
		end
		err[k] /= N
	end

	return err
end

# ╔═╡ 9dbbc5cb-133d-4d20-b6a3-77f4b2ef60ae
md"Compute the error using varying $k$ values using the ROM algorithm."

# ╔═╡ b96344c7-61fc-4cac-8b69-7bc831cbfea1
err = do_experiment(λ_train, λ_test, ds, d′s)

# ╔═╡ 3243a338-327a-4b4b-87fb-cbdbff7e2d7f
md"Compute the error using varying $k$ values using gridded interpolation."

# ╔═╡ 4b5b6ad2-ade9-40eb-9167-61c41b48b5e0
"""
	do_interpolate(λ_train, λ_test, ds)

Creates an `interpolation` object using linear gridded interpolation which is then used on `λ_test`.
"""
function do_interpolate(λ_train, λ_test, ds)
	indices = sortperm(λ_train)
	itp = extrapolate(interpolate((λ_train[indices],), ds[indices], Gridded(Linear())), Line())

	return itp.(λ_test)
end
	

# ╔═╡ 541b29a9-7366-4d03-beb6-3c3287d3e8ae
"""
	do_experiment_interp(λ_train, λ_test, ds, d′s, interp_func::Function)

Do the experiment, but with interpolation instead of a ROM.
"""
function do_experiment_interp(λ_train, λ_test, ds, d′s, interp_func::Function)
	N = length(λ_train)
	err = zeros(N)
	for k in 1:N
		try
			d̂s = interp_func(λ_train[1:k], λ_test, ds[1:k])
			for (i, d̂) in enumerate(d̂s)
				err[k] += norm(d̂, get_data(λ_test[i])[1])
			end
			err[k] /= N
		catch
			err[k] = NaN
		end
	end

	return err
end

# ╔═╡ a5600486-a3a8-4450-a995-14b753ebf838
err_interp = do_experiment_interp(λ_train, λ_test, ds, d′s, do_interpolate)

# ╔═╡ 3b8d1722-f49d-4ec8-8448-c7f3143020a0
md"Plot of the error as a function of $k$. Error converges supralinearly, however too many data points (increasing $k$) results in instability ($k > 40$). We see that we achieve acceptable accuracy using only a few data points as 'training data'."

# ╔═╡ e68313f1-fff1-48bc-8ad3-8cc131297553
begin
	scatter(err, xlabel="k", ylabel="error", title="convergence of algorithm as a function of k", yscale=:log10, label="ROM")
	scatter!(err_interp, label="interp")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Interpolations = "~0.13.6"
Plots = "~1.27.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "af237c08bda486b74318c8070adb96efa6952530"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cd6efcf9dc746b06709df14e462f0a3fe0786b1e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.2+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "b7bc05649af456efc75d178846f47006c2c4c3c7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.6"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3b429f37de37f1fc603cc1de4a799dc7fbe4c0b6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "6f2dd1cf7a4bbf4f305a0d8750e351cb46dfbe80"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.6"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "cd56bf18ed715e8b09f06ef8c6b781e6cdc49911"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─fc4be02c-c27c-11ec-1c65-d5400975275e
# ╠═dd578ceb-8038-4967-ac01-6fb337b3328b
# ╟─4a9bfc40-d165-4fa5-8b36-e1db08d98249
# ╟─5b7b9d6f-f244-41d5-b4c9-675470aa7df7
# ╟─9cfae88e-f6f3-4364-816e-8aad0f1a1033
# ╟─b2807c3a-bd5a-443f-afde-0fc8a340540a
# ╟─0613a3f2-4d5d-4b3b-87a4-f9695f2ec43c
# ╟─def01aeb-7f2c-431f-b391-e9209d30008d
# ╟─4275508e-9180-48da-99d7-4a350b6f1d5a
# ╟─85d1c4af-d4ae-4f17-a4f1-f39cb0ad3f3a
# ╟─e87ac081-5a5b-4886-a8bd-0a0ef4c820e8
# ╟─65646978-f09c-47d1-a60e-3b3aca8b54aa
# ╟─cb698a86-4499-4189-85cd-326711efffa3
# ╟─8281f11f-3b6b-4b8c-91a9-274ba6d29690
# ╟─46a87084-e158-4fa7-b335-946b2d76a689
# ╟─0e4bbebb-87dc-46c6-afce-41a77d6cd821
# ╠═1579cb7d-fd21-4ca1-8ca1-111e08ba654a
# ╠═388c4deb-4be6-4dbe-b0fa-b1f8bb20b418
# ╠═538f2513-7ca0-493d-8269-9a53c55e4832
# ╠═4f6f61f3-32cc-4aa0-8142-979cb55b0274
# ╠═c68cea83-33a6-4cac-a88d-5e8f7d68c4ed
# ╠═4dce59ce-8d70-4588-b5a5-856524c79c88
# ╠═94ac5af7-2c5d-4d5b-ba9b-6a215a95401b
# ╠═57d63ae6-2032-43a3-8333-0216e0e6f45d
# ╠═e2e40cdc-6806-4648-8c0e-562ec2238840
# ╠═6662e106-5309-4d0a-ad65-c5f13b1e6705
# ╠═26f4949d-b2b7-4e3e-a311-f287210aa7b8
# ╠═49437688-2398-4f41-92d7-b4ae50dc413e
# ╠═f77d569d-b209-4f2a-9ced-0ce144e26b2f
# ╠═0aa19f8b-be62-49e5-9942-61f798688e3c
# ╠═c68c27ca-f3a6-408f-8d6b-d50c51bde1d1
# ╠═cb08c667-fb53-4688-bbfe-c9c045a82a1b
# ╠═b5d1385c-066b-4a06-9815-c75b5f4767ff
# ╠═9a9dd256-f669-4397-aff3-a7f37055e637
# ╠═bb4f009b-6b87-4c5a-a727-be36956a33a3
# ╠═495da75a-cf51-4424-ab48-54bead162ce8
# ╠═8d8fb253-1e4d-4786-86ff-0c195a61f0e8
# ╠═9dbbc5cb-133d-4d20-b6a3-77f4b2ef60ae
# ╠═b96344c7-61fc-4cac-8b69-7bc831cbfea1
# ╠═3243a338-327a-4b4b-87fb-cbdbff7e2d7f
# ╠═4b5b6ad2-ade9-40eb-9167-61c41b48b5e0
# ╠═541b29a9-7366-4d03-beb6-3c3287d3e8ae
# ╠═a5600486-a3a8-4450-a995-14b753ebf838
# ╟─3b8d1722-f49d-4ec8-8448-c7f3143020a0
# ╠═e68313f1-fff1-48bc-8ad3-8cc131297553
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
