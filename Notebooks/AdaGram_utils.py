"""
AdaGram

Adaptive Skip-gram (AdaGram) model is a nonparametric extension of famous Skip-gram model implemented in word2vec software which is able to learn multiple representations per word capturing different word meanings. This projects implements AdaGram in Julia language.


https://github.com/sbos/AdaGram.jl/blob/master/src/util.jl
---
function nearest_neighbors(vm::VectorModel, dict::Dictionary, word::DenseArray{Tsf},
		K::Integer=10; exclude::Array{Tuple{Int32, Int64}}=Array(Tuple{Int32, Int64}, 0),
		min_count::Float64=1.)
	sim = zeros(Tsf, (T(vm), V(vm)))

	for v in 1:V(vm)
		for s in 1:T(vm)
			if vm.counts[s, v] < min_count
				sim[s, v] = -Inf
				continue
			end
			in_vs = view(vm.In, :, s, v)
			sim[s, v] = dot(in_vs, word) / norm(in_vs)
		end
	end
	for (v, s) in exclude
		sim[s, v] = -Inf
	end
	top = Array(Tuple{Int, Int}, K)
	topSim = zeros(Tsf, K)

	function split_index(sim, i)
		i -= 1
		v = i % size(sim, 1) + 1
		s = Int(floor(i / size(sim, 1))) + 1
		return v, s
	end
	for k in 1:K
		curr_max = split_index(sim, indmax(sim))
		topSim[k] = sim[curr_max[1], curr_max[2]]
		sim[curr_max[1], curr_max[2]] = -Inf

		top[k] = curr_max
	end
	return Tuple{AbstractString, Int, Tsf}[(dict.id2word[r[2]], r[1], simr)
		for (r, simr) in zip(top, topSim)]
end

function nearest_neighbors(vm::VectorModel, dict::Dictionary,
		w::AbstractString, s::Int, K::Integer=10)
	v = dict.word2id[w]
	return nearest_neighbors(vm, dict, vec(vm, v, s), K; exclude=[(v, s)])
end

cos_dist(x, y) = 1. - dot(x, y) / norm(x, 2) / norm(y, 2)

function disambiguate{Tw <: Integer}(vm::VectorModel, x::Tw,
		context::AbstractArray{Tw, 1}, use_prior::Bool=true,
		min_prob::Float64=1e-3)
	z = zeros(T(vm))

	if use_prior
		expected_pi!(z, vm, x)
		for k in 1:T(vm)
			if z[k] < min_prob
				z[k] = 0.
			end
			z[k] = log(z[k])
		end
	end
	for y in context
		var_update_z!(vm, x, y, z)
	end

	exp_normalize!(z)
	
	return z
end

function disambiguate{Ts <: AbstractString}(vm::VectorModel, dict::Dictionary, x::AbstractString, context::AbstractArray{Ts, 1}, use_prior::Bool=true, min_prob::Float64=1e-3)
	return disambiguate(vm, dict.word2id[x], Int32[dict.word2id[y] for y in context], use_prior, min_prob)
end
"""