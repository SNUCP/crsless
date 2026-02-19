struct cGSWKey
    val::Vector{RLWEkey}
end

ternary_cgswkey(us::UniformSampler, N::Int64, k::Int64, hw::Int64=0) = cGSWKey(RLWEkey[ternary_ringkey(us, N, hw) for _ = 1 : k])

const cGSWkeyPQ = Tensor

cGSWkeyPQ(key::cGSWKey, eval::PolyEvaluatorRNS) = begin
    k = length(key.val)
    Tensor([RLWEkeyPQ(key.val[i], eval) for i = 1 : k])
end