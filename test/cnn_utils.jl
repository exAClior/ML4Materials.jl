@testset "utils" begin
    @test _pad_1d([1, 2, 3], 1) == [0, 1, 2, 3, 0]

end
