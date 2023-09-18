module ML4Materials

using DSP, LinearAlgebra, Enzyme

# Write your package code here.
include("cnn.jl")
include("cnn_utils.jl")
export conv, _pad_1d, conv_sum

end
