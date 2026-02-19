include("src/Mkfhe.jl")
import HIENAA.Math: SimpleScaler, simple_scale_to!

global hw = 192
global recommended_sketch = Dict{Int64,MKParamSketch}(
    # Since each Qi needs to be small, we set logQ = 100. 
    # This results in logQ = 105 when code is run.
    2 => MKParamSketch(2, CyclotomicParam(1 << 13), 100, 6),
    4 => MKParamSketch(4, CyclotomicParam(1 << 14), 214, 5),
    8 => MKParamSketch(8, CyclotomicParam(1 << 14), 214, 6),
)

function vector_gadgetprod_test(test_id::Int64)
    #######################################################################
    ############################# SETUP ###################################
    #######################################################################

    sketch = recommended_sketch[test_id]
    param = MKParameters(sketch)
    N, Qlen, K = param.ring_param.N, length(param.Q), param.max_parties

    #######################################################################
    ############################ ENCRYPTION ###############################
    #######################################################################

    oper = MKOperator(param)
    key = ternary_cgswkey(UniformSampler(), N, K, hw)
    entor = MKEncryptor(key, 3.2, oper)

    ptxt_modulus = 256

    msg = rand(0:ptxt_modulus-1)
    pt = PlainConst(ModScalar(msg, oper.operQ.evalQ))
    ct = clev_encrypt(pt, K, entor)

    #######################################################################
    ############################ COMPUTATION ###############################
    #######################################################################

    v = Tensor(N, Qlen, K, isntt=false)
    res = Tensor(N, Qlen, K + 1)

    sc1 = SimpleScaler([Modulus(ptxt_modulus)], oper.operQ.evalQ.moduli)
    for i = 1:K
        @. v[i].coeffs[1] = i
        simple_scale_to!(v[i].coeffs, v[i].coeffs[1:1], sc1)
    end

    GC.gc()
    print("Vector Gadget Product Took :           ")
    @time vector_gadgetprod_to!(res, v, ct, oper)

    #######################################################################
    ############################ CHECKING ###############################
    #######################################################################

    sc2 = SimpleScaler(oper.operQ.evalQ.moduli, [Modulus(ptxt_modulus)])
    p = phase(res, entor)
    for i = 1:K
        p[i].isntt[] && intt_to!(p[i], p[i], oper.operQ.evalQ)
        simple_scale_to!(p[i].coeffs[1:1], p[i].coeffs, sc2)

        for j = 1:N
            @assert p[i].coeffs[1][j] == (msg * i) % ptxt_modulus "FAIL: VECTOR GADGET PRODUCT // TESTID $test_id"
        end
    end

    nothing
end

function mk_vector_gadgetprod_test(test_id::Int64)
    #######################################################################
    ############################# SETUP ###################################
    #######################################################################

    sketch = recommended_sketch[test_id]
    param = MKParameters(sketch)
    N, Qlen, K = param.ring_param.N, length(param.Q), param.max_parties

    #######################################################################
    ############################ ENCRYPTION ###############################
    #######################################################################

    ptxt_modulus = 256

    oper = MKOperator(param)
    keys = [ternary_cgswkey(UniformSampler(), N, K, hw) for _ = 1:K]
    entors = [MKEncryptor(key, 3.2, oper) for key in keys]

    msg = [rand(0:ptxt_modulus-1) for _ = 1:K]
    pts = [PlainConst(ModScalar(msg[i], oper.operQ.evalQ)) for i = 1:K]
    cts = [clev_encrypt(pts[i], K, entors[i]) for i = 1:K]

    v = Vector{Tensor}(undef, K + 1)
    for i = 1:K+1
        v[i] = Tensor(N, Qlen, i)
    end

    v[1][1].isntt[] = false
    @. v[1][1].coeffs[1] = 1
    sc1 = SimpleScaler([Modulus(ptxt_modulus)], oper.operQ.evalQ.moduli)
    simple_scale_to!(v[1][1].coeffs, v[1][1].coeffs[1:1], sc1)

    #######################################################################
    ############################ COMPUTATION ###############################
    #######################################################################

    GC.gc()
    print("Multi-Key Vector Gadget Product Took : ")
    @time begin
        for i = 1:K
            shrink!(cts[i], i)
            vector_gadgetprod_to!(v[i+1], v[i], cts[i], oper)
        end
    end

    #######################################################################
    ############################ CHECKING ###############################
    #######################################################################

    res = ModPoly(N, Qlen)
    jd = JointDecryptor(keys, oper)
    joint_decrypt_to!(res, v[K+1], jd)

    res.isntt[] && intt_to!(res, res, oper.operQ.evalQ)
    sc2 = SimpleScaler(oper.operQ.evalQ.moduli, [Modulus(ptxt_modulus)])
    simple_scale_to!(res.coeffs[1:1], res.coeffs, sc2)

    for i = 1:N
        @assert res.coeffs[1][i] == prod(msg) % ptxt_modulus "FAIL: MULTI-KEY VECTOR GADGET PRODUCT // TESTID $test_id"
    end

    nothing
end

function rgsw_extend_test(test_id::Int64)
    #######################################################################
    ############################# SETUP ###################################
    #######################################################################

    sketch = recommended_sketch[test_id]
    param = MKParameters(sketch)
    N, Qlen, K = param.ring_param.N, length(param.Q), param.max_parties

    #######################################################################
    ############################ ENCRYPTION ###############################
    #######################################################################

    oper = MKOperator(param)
    keys = [ternary_cgswkey(UniformSampler(), N, K, hw) for _ = 1:K]
    entors = [MKEncryptor(key, 3.2, oper) for key in keys]

    evalQ = oper.operQ.evalQ
    ptxt_modulus = 256
    msgs = [rand(0:ptxt_modulus-1) for _ = 1:K]
    cts = [cgsw_encrypt(PlainConst(ModScalar(msgs[i], evalQ)), K, entors[i]) for i = 1:K]
    evks = [cgsw_encrypt(PlainConst(ModScalar(1, evalQ)), K, entors[i]) for i = 1:K]

    #######################################################################
    ############################ COMPUTATION ###############################
    #######################################################################

    glen = oper.operQ.decer.glen
    mgsw = [MGSW(N, K, Qlen, glen) for _ = 1:K]

    for i = 1:K
        shrink!(cts[i], i)
        shrink!(evks[i], i)
    end

    for i = 1:K-1
        mgsw_extend!(mgsw[i], cts[i], i, evks, oper)
    end

    GC.gc()
    print("RGSW Extension Took :                  ")
    @time mgsw_extend!(mgsw[K], cts[K], K, evks, oper)

    #######################################################################
    ############################ CHECKING ###############################
    #######################################################################

    p = ModPoly(N, Qlen)
    jd = JointDecryptor(keys, oper)

    evalQ = oper.operQ.evalQ
    sc = Vector{SimpleScaler}(undef, glen)
    for i = 1:glen
        sc[i] = SimpleScaler(evalQ.moduli, evalQ.moduli[i:i])
    end

    for idx = 1:K
        for j = 1:glen
            joint_decrypt_to!(p, mgsw[idx].basket[1].stack[j], jd)
            p.isntt[] && intt_to!(p, p, evalQ)
            simple_scale_to!(p.coeffs[1:1], p.coeffs, sc[j])

            for k = 1:glen
                j == k && continue
                mul_to!(p.coeffs[1], evalQ.moduli[k].Q, p.coeffs[1], evalQ[j])
            end

            @assert p.coeffs[1][1] == msgs[idx] % ptxt_modulus "FAIL: RGSW EXTENSION // TESTID $test_id"
            for k = 2:N
                @assert p.coeffs[1][k] == 0 "FAIL: RGSW EXTENSION // TESTID $test_id"
            end

            msg = p.coeffs[1][1]

            for i = 1:K
                joint_decrypt_to!(p, mgsw[idx].basket[i+1].stack[j], jd)
                p.isntt[] && intt_to!(p, p, evalQ)
                simple_scale_to!(p.coeffs[1:1], p.coeffs, sc[j])

                for k = 1:glen
                    j == k && continue
                    mul_to!(p.coeffs[1], evalQ.moduli[k].Q, p.coeffs[1], evalQ[j])
                end

                intt_to!(p.coeffs[2], jd.jointkey[i][j], evalQ[j])
                mulsub_to!(p.coeffs[1], msg, p.coeffs[2], evalQ[j])

                for k = 1:N
                    @assert p.coeffs[1][k] == 0 "FAIL: RGSW EXTENSION // TESTID $test_id"
                end
            end
        end
    end

    nothing
end

function rgsw_internal_product_test(test_id::Int64)
    #######################################################################
    ############################# SETUP ###################################
    #######################################################################

    sketch = recommended_sketch[test_id]
    param = MKParameters(sketch)
    N, Qlen, K = param.ring_param.N, length(param.Q), param.max_parties

    oper = MKOperator(param)
    keys = [ternary_cgswkey(UniformSampler(), N, K, hw) for _ = 1:K]
    entors = [MKEncryptor(key, 3.2, oper) for key in keys]

    #######################################################################
    ############################ ENCRYPTION ###############################
    #######################################################################

    evalQ = oper.operQ.evalQ
    msgs = [rand([0, 1]) for _ = 1:K]
    cts = [cgsw_encrypt(PlainConst(ModScalar(msgs[i], evalQ)), K, entors[i]) for i = 1:K]
    evks = [cgsw_encrypt(PlainConst(ModScalar(1, evalQ)), K, entors[i]) for i = 1:K]

    for i = 1:K
        shrink!(cts[i], i)
        shrink!(evks[i], i)
    end

    glen = oper.operQ.decer.glen
    mgsw = [MGSW(N, K, Qlen, glen) for _ = 1:K]
    res = MGSW(N, K, Qlen, glen)
    for i = 1:K
        mgsw_extend!(mgsw[i], cts[i], i, evks, oper)
    end

    idx1, idx2 = rand(1:K), rand(1:K)
    while idx1 != idx2
        idx1, idx2 = rand(1:K), rand(1:K)
    end

    GC.gc()
    print("RGSW Internal Product Took :           ")
    @time internal_product_to!(res, mgsw[idx1], mgsw[idx2], oper)

    #######################################################################
    ############################ CHECKING ###############################
    #######################################################################

    p = ModPoly(N, Qlen)
    jd = JointDecryptor(keys, oper)

    evalQ = oper.operQ.evalQ
    sc = Vector{SimpleScaler}(undef, glen)
    for i = 1:glen
        sc[i] = SimpleScaler(evalQ.moduli, evalQ.moduli[i:i])
    end

    for j = 1:glen
        joint_decrypt_to!(p, res.basket[1].stack[j], jd)
        p.isntt[] && intt_to!(p, p, evalQ)
        simple_scale_to!(p.coeffs[1:1], p.coeffs, sc[j])

        for k = 1:glen
            j == k && continue
            mul_to!(p.coeffs[1], evalQ.moduli[k].Q, p.coeffs[1], evalQ[j])
        end

        @assert p.coeffs[1][1] == msgs[idx1] * msgs[idx2] "FAIL: RGSW EXTENSION // TESTID $test_id"
        for k = 2:N
            @assert p.coeffs[1][k] == 0 "FAIL: RGSW EXTENSION // TESTID $test_id"
        end

        msg = p.coeffs[1][1]

        for i = 1:K
            joint_decrypt_to!(p, res.basket[i+1].stack[j], jd)
            p.isntt[] && intt_to!(p, p, evalQ)
            simple_scale_to!(p.coeffs[1:1], p.coeffs, sc[j])

            for k = 1:glen
                j == k && continue
                mul_to!(p.coeffs[1], evalQ.moduli[k].Q, p.coeffs[1], evalQ[j])
            end

            intt_to!(p.coeffs[2], jd.jointkey[i][j], evalQ[j])
            mulsub_to!(p.coeffs[1], msg, p.coeffs[2], evalQ[j])

            for k = 1:N
                @assert p.coeffs[1][k] == 0 "FAIL: RGSW EXTENSION // TESTID $test_id"
            end
        end
    end

    nothing
end

test_id = 2
vector_gadgetprod_test(test_id)
mk_vector_gadgetprod_test(test_id)
rgsw_extend_test(test_id)
rgsw_internal_product_test(test_id)