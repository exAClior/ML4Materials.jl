function conv(input::Vector{T}, kernel::Vector{T}) where {T<:Number}
    paded_input = _pad_1d(input, length(kernel) ÷ 2)
    output = zeros(T, length(input))
    for i = 1:length(input)
        output[i] = sum(view(paded_input, i:i+length(kernel)-1) .* kernel)
    end
    return output
end
