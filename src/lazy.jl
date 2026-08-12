# `@lazy` is deliberately a syntactic, lexical opt-in.  Its only rewrites are
# contractions; structural operations propagate laziness from their inputs.
@inline _lazy_finalize(q::AbstractTLArray) = TLArray(q)
@inline _lazy_finalize(x::Number) = x
_lazy_finalize(x) = throw(ArgumentError(
    "@lazy must finish with an AbstractTLArray or Number, got $(typeof(x))"))

@inline _lazy_callee(name::Symbol) =
    name === :contract || name === :* ? GlobalRef(@__MODULE__, :_lazy_contract) : name

@inline function _lazy_callee(callee::Expr)
    callee.head === :. || return callee
    length(callee.args) == 2 || return callee
    callee.args[1] === :Telum || return callee
    name = callee.args[2]
    name isa QuoteNode || return callee
    name.value === :contract || return callee
    return GlobalRef(@__MODULE__, :_lazy_contract)
end

_lazy_rewrite(ex) = _lazy_rewrite(ex, false)
_lazy_rewrite(ex, ::Bool) = ex

function _lazy_rewrite(ex::Expr, in_nested_function::Bool)
    ex.head === :quote && return ex
    ex.head === :inert && return ex
    ex.head === :macrocall && return ex
    if ex.head === :(=)
        return Expr(:(=), ex.args[1], _lazy_rewrite(ex.args[2], in_nested_function))
    elseif ex.head === :function || ex.head === :->
        args = copy(ex.args)
        args[end] = _lazy_rewrite(args[end], true)
        return Expr(ex.head, args...)
    elseif ex.head === :return && !in_nested_function
        value = length(ex.args) == 1 ? _lazy_rewrite(ex.args[1], false) : nothing
        return Expr(:return, Expr(:call, GlobalRef(@__MODULE__, :_lazy_finalize), value))
    elseif ex.head === :call
        callee = _lazy_callee(ex.args[1])
        args = map(arg -> _lazy_rewrite(arg, in_nested_function), ex.args[2:end])
        # Julia lowers `a * b * c` to one n-ary call; lazy contractions remain binary.
        if ex.args[1] === :* && length(args) > 2
            return foldl(args[2:end]; init=args[1]) do lhs, rhs
                Expr(:call, callee, lhs, rhs)
            end
        end
        return Expr(:call, callee, args...)
    end
    return Expr(ex.head, map(arg -> _lazy_rewrite(arg, in_nested_function), ex.args)...)
end

macro lazy(ex)
    rewritten = _lazy_rewrite(ex)
    return Expr(:call, GlobalRef(@__MODULE__, :_lazy_finalize), esc(rewritten))
end
