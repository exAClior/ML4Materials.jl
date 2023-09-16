function _pad_1d(x::Vector{T}, pad::Int) where {T <: Number}
    return [zeros(pad); x; zeros(pad)]
end
