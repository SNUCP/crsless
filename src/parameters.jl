struct MKParamSketch
    max_parties::Int64
    ring_param::RingParam
    logQ::Int64
    Qlen::Int64

    MKParamSketch(max_parties::Int64, ring_param::RingParam, logQ::Int64, Qlen::Int64=0) = begin
        if Qlen == 0
            Qlen = ceil(Int64, logQ / 62)
        end
        
        new(max_parties, ring_param, logQ, Qlen)
    end
end

struct MKParameters
    max_parties::Int64
    ring_param::RingParam
    Q::Vector{UInt64}
    dlen::Int64

    MKParameters(max_parties::Int64, ring_param::RingParam, Q::Vector{UInt64}) =
        new(max_parties, ring_param, Q, 1)

    function MKParameters(sketch::MKParamSketch)::MKParameters
        max_parties, ring_param, logQ, Qlen = sketch.max_parties, sketch.ring_param, sketch.logQ, sketch.Qlen

        Qbits = logQ / Qlen
        Qprimes = find_prime(ring_param, Qbits, Qlen)
        Q = Qprimes

        new(max_parties, ring_param, Q, 1)
    end
end