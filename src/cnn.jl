function conv(input::Vector{T}, kernel::Vector{M}) where {T<:Number,M<:Number}
    paded_input = _pad_1d(input, length(kernel) ÷ 2)
    output = zeros(promote_type(T,M), length(input))
    for i = 1:length(input)
        output[i] = sum(view(paded_input, i:i+length(kernel)-1) .* kernel)
    end
    return output
end

function conv_sum(input::Vector{T}, kernel::Vector{M}) where {T<:Number, M<:Number}
    output = conv(input, kernel)
    return sum(output)
end
