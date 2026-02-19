struct cLEV
    glen::Int64
    stack::Vector{Tensor}

    function cLEV(N::Int64, k::Int64, len::Int64, glen::Int64; isntt::Bool=false)
        new(glen, Tensor[Tensor(N, len, k+1, isntt=isntt) for _ = 1 : glen * k])
    end
end

function shrink!(c::cLEV, k::Int64)
    resize!(c.stack, c.glen * k)

    K, Qlen = length(c.stack[1].vals)-1, length(c.stack[1][1])
    for i = 1 : c.glen * k
        copy!(c.stack[i][k+1], c.stack[i][K+1])
        c.stack[i] = c.stack[i][1:k+1, 1:Qlen]
    end
end

struct cGSW
    basket::Vector{cLEV}

    function cGSW(N::Int64, k::Int64, len::Int64, glen::Int64; isntt::Bool=false)
        new(cLEV[cLEV(N, k, len, glen, isntt=isntt) for _ = 1 : k+1])
    end
end

function shrink!(c::cGSW, k::Int64)    
    resize!(c.basket, k+1)
    for i = 1 : k+1
        shrink!(c.basket[i], k)
    end
end

struct MLEV
    glen::Int64
    stack::Vector{Tensor}

    function MLEV(N::Int64, k::Int64, len::Int64, glen::Int64; isntt::Bool=false)
        new(glen, Tensor[Tensor(N, len, k+1, isntt=isntt) for _ = 1 : glen])
    end
end

struct MGSW
    basket::Vector{MLEV}

    function MGSW(N::Int64, k::Int64, len::Int64, glen::Int64; isntt::Bool=false)
        new([MLEV(N, k, len, glen, isntt=isntt) for _ = 1 : k+1])
    end
end