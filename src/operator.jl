struct MKOperator
    operQ::Operator
    ct_buff::Vector{Tensor}
    tensor_buff::Vector{Tensor}

    function MKOperator(param::MKParameters)::MKOperator
        K, ring_param, Q, dlen = param.max_parties, param.ring_param, param.Q, param.dlen
        N, Qlen = ring_param.N, length(Q)

        rlwe_param = RLWEParameters(ring_param, missing, Q, dlen)
        operQ = Operator(rlwe_param)

        ct_buff = [Tensor(N, Qlen, K + 1, isntt=false) for _ = 1:2]
        tensor_buff = [Tensor(N, Qlen, operQ.decer.glen) for _ = 1:K+2]

        new(operQ, ct_buff, tensor_buff)
    end
end

function hoisted_vector_gadgetprod_to!(res::Tensor, decv::Vector{Tensor}, ct::cLEV, oper::MKOperator; zeroidx::Int64=typemax(Int64))
    k, len = size(ct.stack[1])
    k -= 1

    @assert length(res.vals) == k + 1 && length(decv) == k

    operQ = oper.operQ
    evalQ = operQ.evalQ

    ct_buff = oper.ct_buff[1][1:k+1, 1:len]
    initialise!(ct_buff, isntt=true, isPQ=false)

    glen = oper.operQ.decer.glen
    for i = 1:min(k, zeroidx)
        for j = 1:glen
            !decv[i][j].isntt[] && ntt_to!(decv[i][j], decv[i][j], evalQ)
            for idx = 1:k+1
                muladd_to!(ct_buff[idx], decv[i][j], ct.stack[(i-1)*glen+j][idx], evalQ)
            end
        end
    end

    copy!(res, ct_buff)

    return nothing
end

function vector_gadgetprod_to!(res::Tensor, v::Tensor, ct::cLEV, oper::MKOperator; zeroidx::Int64=typemax(Int64))
    k, len = size(v)
    operQ = oper.operQ
    decer = operQ.decer
    glen = operQ.decer.glen

    buff = oper.tensor_buff[k+2][1, 1:len]
    decbuff = [oper.tensor_buff[i][1:glen, 1:len] for i = 1:k]

    for i = 1:min(k, zeroidx)
        copy!(buff, v[i])
        buff.isntt[] && intt_to!(buff, buff, operQ.evalQ)
        decompose_to!(decbuff[i], PlainPoly(buff), decer)
    end

    hoisted_vector_gadgetprod_to!(res, decbuff, ct, oper; zeroidx=zeroidx)

    return nothing
end

function mgsw_extend!(res::MGSW, ct::cGSW, idx::Int64, evk::Vector{cGSW}, oper::MKOperator)
    n = length(evk)
    operQ = oper.operQ
    evalQ = operQ.evalQ
    glen = operQ.decer.glen
    Qlen = length(evalQ)

    for i = 1:glen
        initialise!(res.basket[1].stack[i])
        add_to!(res.basket[1].stack[i][1], res.basket[1].stack[i][1], operQ.decer.gvec[i], evalQ)
    end

    for i = 1:n, j = 1:glen
        if i != idx
            initialise!(res.basket[i+1].stack[j])

            acci = res.basket[i+1].stack[j][1:i+1, 1:Qlen]
            tmp = oper.ct_buff[2][1:i+1, 1:Qlen]
            for k = 1:i
                acctmp = res.basket[k].stack[j][1:i, 1:Qlen]
                vector_gadgetprod_to!(tmp, acctmp, evk[i].basket[k+1], oper, zeroidx = i > idx ? max(k, idx+1) : k)
                add_to!(acci, acci, tmp, operQ)
            end
        else
            initialise!(res.basket[i+1].stack[j])

            acci = res.basket[i+1].stack[j][1:i+1, 1:Qlen]
            tmp = oper.ct_buff[2][1:i+1, 1:Qlen]

            decer = operQ.decer
            buff = oper.tensor_buff[i+1][1, 1:Qlen]
            decbuff = [oper.tensor_buff[ℓ][1:glen, 1:Qlen] for ℓ = 1:i]

            for k = 1:i
                acckprev = res.basket[k].stack[j][1:i, 1:Qlen]
                accknext = res.basket[k].stack[j][1:i+1, 1:Qlen]

                for ℓ = 1:k
                    copy!(buff, acckprev[ℓ])
                    buff.isntt[] && intt_to!(buff, buff, operQ.evalQ)
                    decompose_to!(decbuff[ℓ], PlainPoly(buff), decer)
                end

                hoisted_vector_gadgetprod_to!(tmp, decbuff, ct.basket[k+1], oper, zeroidx=k)
                add_to!(acci, acci, tmp, operQ)

                vector_gadgetprod_to!(accknext, acckprev, ct.basket[1], oper, zeroidx=k)
            end
        end
    end
end

function hoisted_external_product_to!(res::Tensor, decxi::Vector{Tensor}, y::MGSW, oper::MKOperator)
    k, len = size(y.basket[1].stack[1])
    k -= 1

    @assert length(res.vals) == k + 1 && length(decxi) == k + 1

    operQ = oper.operQ
    evalQ = operQ.evalQ

    ct_buff = oper.ct_buff[1][1:k+1, 1:len]
    initialise!(ct_buff, isntt=true, isPQ=false)

    glen = operQ.decer.glen
    for i = 1:k+1
        for j = 1:glen
            !decxi[i][j].isntt[] && ntt_to!(decxi[i][j], decxi[i][j], evalQ)
            for idx = 1:k+1
                muladd_to!(ct_buff[idx], decxi[i][j], y.basket[i].stack[j][idx], evalQ)
            end
        end
    end

    copy!(res, ct_buff)

    return nothing
end

# h(x)^T * y
function internal_product_to!(res::MGSW, x::MGSW, y::MGSW, oper::MKOperator)
    k, len = size(x.basket[1].stack[1])
    k -= 1

    @assert length(res.basket) == k + 1 && length(x.basket) == k + 1 && length(y.basket) == k + 1

    operQ = oper.operQ
    evalQ, decer = operQ.evalQ, operQ.decer
    glen = operQ.decer.glen
    
    buff = oper.tensor_buff[k+2][1, 1:len]
    decbuff = [oper.tensor_buff[i][1:glen, 1:len] for i = 1:k+1]

    for i = 1:k+1, j = 1:glen
        for idx = 1:k+1
            copy!(buff, x.basket[i].stack[j][idx])
            buff.isntt[] && intt_to!(buff, buff, evalQ)
            decompose_to!(decbuff[idx], PlainPoly(buff), decer)
        end

        hoisted_external_product_to!(res.basket[i].stack[j], decbuff, y, oper)
    end

    return nothing
end