struct MKEncryptor
    key::cGSWKey
    keyPQ::cGSWkeyPQ
    usampler::UniformSampler
    gsampler::CDTSampler
    oper::Operator

    MKEncryptor(key::cGSWKey, σ::Real, oper::MKOperator)::MKEncryptor = begin
        operQ = oper.operQ
        evalQ = operQ.evalQ

        keyPQ = cGSWkeyPQ(key, evalQ)
        for i = eachindex(keyPQ.vals)
            ntt!(keyPQ[i], evalQ)
        end

        new(key, keyPQ, UniformSampler(), CDTSampler(0.0, σ), operQ)
    end
end

function salwe_sample(K::Int64, entor::MKEncryptor)
    oper, N = entor.oper, entor.keyPQ[1].N
    len = length(oper.evalQ)
    res = Tensor(N, len, K + 1)
    salwe_sample_to!(res, entor)
    res
end

function salwe_sample_to!(res::Tensor, entor::MKEncryptor)::Nothing
    us, gs = entor.usampler, entor.gsampler
    K, len = size(res)
    K -= 1
    N = res[1].N

    key = entor.keyPQ
    eval = entor.oper.evalQ

    for i = 1:K
        res[i].isntt[] = false
    end

    res[K+1].isntt[] = true
    uniform_random_to!(us, res[K+1], eval)

    for idx = 1:K
        @inbounds for j = 1:N
            ei = sample(gs)
            for i = 1:len
                res[idx].coeffs[i][j] = Bred(ei, eval[i])
            end
        end
    end

    for i = 1:K
        ntt!(res[i], eval)
        mulsub_to!(res[i], res[K+1], key[i], eval)
    end

    return nothing
end

function phase(ct::Tensor, entor::MKEncryptor)
    K, _ = size(ct)
    res = Tensor(ct.vals[1].N, length(ct.vals[1]), K-1)
    phase_to!(res, ct, entor)
    res
end

function phase_to!(res::Tensor, ct::Tensor, entor::MKEncryptor)
    len = length(ct[1])
    buff = entor.oper.tensor_buff[end, 1:len]
    key = entor.keyPQ
    eval = entor.oper.evalQ
    K = length(ct.vals) - 1

    copy!(buff, ct[K+1])
    !buff.isntt[] && ntt!(buff, eval)

    for i = 1:K
        copy!(res[i], ct[i])
        !res[i].isntt[] && ntt!(res[i], eval)
        muladd_to!(res[i], buff, key[i], eval)
        intt!(res[i], eval)
    end

    return nothing
end

function clev_encrypt(m::PlainText, k::Int64, entor::MKEncryptor)
    len, glen, N = length(entor.oper.evalQ), entor.oper.decer.glen, entor.keyPQ[1].N
    res = cLEV(N, k, len, glen)
    clev_encrypt_to!(res, m, entor)
    res
end

function clev_encrypt_to!(res::cLEV, m::PlainConst, entor::MKEncryptor)::Nothing
    glen = entor.oper.decer.glen
    K = length(res.stack) ÷ glen
    eval = entor.oper.evalQ

    for i = 1:K
        for j = 1:glen
            salwe_sample_to!(res.stack[(i-1)*glen+j], entor)
            muladd_to!(res.stack[(i-1)*glen+j][i], entor.oper.decer.gvec[j], m.val, eval)
        end
    end
end

function clev_encrypt_to!(res::cLEV, m::PlainPoly, entor::MKEncryptor)::Nothing
    glen = entor.oper.decer.glen
    k = length(res.stack) ÷ glen
    eval = entor.oper.evalQ
    buff = entor.oper.tensor_buff[1, 1:length(eval)]

    copy!(buff, m.val)
    !buff.isntt[] && ntt_to!(buff, buff, eval)

    for i = 1:k
        for j = 1:glen
            salwe_sample_to!(res.stack[(i-1)*glen+j], entor)
            muladd_to!(res.stack[(i-1)*glen+j][i], entor.oper.decer.gvec[j], buff, eval)
        end
    end
end

function cgsw_encrypt(m::PlainText, k::Int64, entor::MKEncryptor)
    len, glen, N = length(entor.oper.evalQ), entor.oper.decer.glen, entor.keyPQ[1].N
    res = cGSW(N, k, len, glen)
    cgsw_encrypt_to!(res, m, entor)
    res
end

function cgsw_encrypt_to!(res::cGSW, m::PlainConst, entor::MKEncryptor)::Nothing
    glen = entor.oper.decer.glen
    k = length(res.basket[1].stack) ÷ glen

    eval = entor.oper.evalQ
    key = entor.keyPQ
    buff = entor.oper.tensor_buff[1, 1:length(eval)]

    for i = 1:k
        for j = 1:glen
            salwe_sample_to!(res.basket[1].stack[(i-1)*glen+j], entor)
            muladd_to!(res.basket[1].stack[(i-1)*glen+j][i], entor.oper.decer.gvec[j], m.val, eval)
        end
    end

    for idx = 1:k
        mul_to!(buff, m.val, key[idx], eval)
        for i = 1:k
            for j = 1:glen
                salwe_sample_to!(res.basket[idx+1].stack[(i-1)*glen+j], entor)
                muladd_to!(res.basket[idx+1].stack[(i-1)*glen+j][i], entor.oper.decer.gvec[j], buff, eval)
            end
        end
    end
end

function cgsw_encrypt_to!(res::cGSW, m::PlainPoly, entor::MKEncryptor)::Nothing
    glen = entor.oper.decer.glen
    k = length(res.basket[1].stack) ÷ glen
    eval = entor.oper.evalQ
    key = entor.keyPQ

    buff1 = entor.oper.tensor_buff[1, 1:length(eval)]
    buff2 = entor.oper.tensor_buff[2, 1:length(eval)]

    copy!(buff1, m.val)
    !buff1.isntt[] && ntt_to!(buff1, buff1, eval)

    for i = 1:k
        for j = 1:glen
            salwe_sample_to!(res.basket[1].stack[(i-1)*glen+j], entor)
            muladd_to!(res.basket[1].stack[(i-1)*glen+j][i], entor.oper.decer.gvec[j], buff1, eval)
        end
    end

    for idx = 1:k
        mul_to!(buff2, m.val, key[idx], eval)
        for i = 1:k
            for j = 1:glen
                salwe_sample_to!(res.basket[idx+1].stack[(i-1)*glen+j], entor)
                muladd_to!(res.basket[idx+1].stack[(i-1)*glen+j][i], entor.oper.decer.gvec[j], buff2, eval)
            end
        end
    end
end

struct JointDecryptor
    keys::Vector{cGSWKey}
    keyPQs::Vector{cGSWkeyPQ}
    jointkey::Vector{ModPoly}
    oper::Operator
    ct_buff::Tensor
    res_buff::Tensor

    JointDecryptor(keys::Vector{cGSWKey}, oper::MKOperator) = begin
        evalQ = oper.operQ.evalQ
        N, Qlen, n = oper.operQ.param.N, length(evalQ), length(keys)
        ct_buff = Tensor(N, Qlen, n+1)
        res_buff = Tensor(N, Qlen, n)

        keyPQs = [cGSWkeyPQ(keys[i], evalQ) for i = 1 : length(keys)]
        for i = eachindex(keyPQs)
            for j = eachindex(keyPQs[i].vals)
                ntt!(keyPQs[i][j], evalQ)
            end
        end

        jointkey = Vector{ModPoly}(undef, n)
        for i = 1:n
            jointkey[i] = ModPoly(N, Qlen, isntt=true)
            add_to!(jointkey[i], jointkey[i], keyPQs[i][1], evalQ)
            for j = 2:i
                muladd_to!(jointkey[i], jointkey[j-1], keyPQs[i][j], evalQ)
            end
        end

        new(keys, keyPQs, jointkey, oper.operQ, ct_buff, res_buff) 
    end
end

function joint_decrypt(ct::Tensor, jd::JointDecryptor)
    res = PlainPoly(ModPoly(jd.oper.operQ.N, length(jd.oper.evalQ)))
    joint_decrypt_to!(res, ct, jd)
    res
end

function joint_decrypt_to!(res::ModPoly, ct::Tensor, jd::JointDecryptor)::Nothing
    evalQ = jd.oper.evalQ
    Qlen = length(evalQ)
    buff = jd.oper.tensor_buff[1, 1:Qlen]

    copy!(res, ct[1])
    !res.isntt[] && ntt!(res, evalQ)

    for i = 1:length(jd.keys)
        copy!(buff, ct[i+1])
        !buff.isntt[] && ntt!(buff, evalQ)
        muladd_to!(res, buff, jd.jointkey[i], evalQ)
    end

    return nothing
end