using ML4Materials
using Test

@testset "ML4Materials.jl" begin
    # Write your tests here.
    include("cnn.jl")
    include("cnn_utils.jl")

end
