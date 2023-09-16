@testset "CNN" begin
    @test conv([1, 2, 3, 4, 5], [1, 1, 1]) == [3, 6, 9, 12, 9]


end
