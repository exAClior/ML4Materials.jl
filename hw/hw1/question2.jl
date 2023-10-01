using Pkg;
Pkg.activate("../..");
using Flux, Random, Plots

"""
    mse(ŷ, y)

Mean squared error loss function
"""
mse(ŷ, y) = sum((ŷ - y) .^ 2) / length(y)

loss(model, x, y) = Flux.crossentropy(model(x), Flux.onehotbatch(y, 0:9))

function high_freq(x)
    return sin(5x) + cos(5x)
end

function low_freq(x)
    return sin(x)
end

function cantor(x, level = 10)
    if level == 0
        return x < 1 // 3 ? 2x : 2x - 1
    elseif x < 1 // 3
        return 0.5cantor(3x, level - 1)
    elseif x < 2 // 3
        return 0.5
    else
        return 0.5 + 0.5cantor(3x - 2, level - 1)
    end
end

function weierstrass(x, a = 0.5, b = 3, n = 25)
    return sum([a^n * cos(b^n * pi * x) for n = 0:n])
end

function periodic_data(x)
    return sin(x)
end

function nonperiodic_data(x)
    return x^2
end

function smooth_data(x)
    return sin(x) + cos(x)
end

function non_smooth_data(x)
    #sawtooth wave
    return x - floor(x)
end

function noisy_data(x)
    return sin(x) + 0.1 * randn()
end

function noiseless_data(x)
    return sin(x)
end

function smooth_vs_non_smooth(max_x, optimizer, num_epochs::Integer)

    func_type = "smooth vs. non-smooth"

    model1 = Chain(Dense(1 => 10, relu), Dense(10 => 1, sigmoid), softmax)
    model2 = Chain(Dense(1 => 10, relu), Dense(10 => 1, sigmoid), softmax)

    op_state_s = Flux.setup(optimizer, model1)
    op_state_ns = Flux.setup(optimizer, model2)

    x_train = rand(1000) .* max_x

    y_s_train = smooth_data.(x_train)
    y_ns_train = non_smooth_data.(x_train)

    data_s = [(x, y) for (x, y) in zip(x_train, y_s_train)]
    data_ns = [(x, y) for (x, y) in zip(x_train, y_ns_train)]


    convergence_smooth = zeros(num_epochs)
    convergence_non_smooth = zeros(num_epochs)
    for ii = 1:num_epochs
        println(ii)
        Flux.train!(model1, data_s, op_state_s) do m, x, y
            loss(m(x), y)
        end

        Flux.train!(model2, data_ns, op_state_ns) do m, x, y
            loss(m(x), y)
        end
    end

    # plot convergence_smooth and convergence_non_smooth in the same plot, label them and save it
    plot(
        convergence_smooth,
        label = "smooth",
        xlabel = "epochs",
        ylabel = "loss",
        title = "smooth vs. non-smooth",
    )
    plot!(convergence_non_smooth, label = "non-smooth")
    savefig("plots/smooth_vs_non_smooth.png")
end



function main()
    # investigate types of target function on fixed neural network on number of epochs
    # smooth vs non-smooth
    # Initialize the ADAM optimizer with default settings
    smooth_vs_non_smooth(5.0, Adam(), 500)
end


main()
