import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


def high_freq(x):
    return np.sin(5*x) + np.cos(5*x)

def low_freq(x):
    return np.sin(x) + np.cos(x)

def cantor(x, level=10):
    if level == 0:
        return 2*x if (x < 1 / 3) else (2*x -1)
    elif x < 1 / 3:
        return cantor(3*x, level-1) / 2
    elif  x < 2 / 3:
        return 1 / 2
    else:
        return cantor(3*x - 2, level-1) / 2 + 1 / 2


def weierstrass(x , a = 0.5, b = 3, n =25):
    return sum([a**i * np.cos(b**i * np.pi * x) for i in range(n)])

def periodic_data(x):
    return np.sin(2 * np.pi * x)

def non_periodic_data(x):
    return x**2

def smooth_data(x):
    return x

def non_smooth_data(x):
    #sawtooth wave
    return x - np.floor(x)

def noisy_data(x):
    return np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, len(x))

def non_noisy_data(x):
    return np.sin(2 * np.pi * x)


def train_loop(x,y,model,num_epochs,learning_rate,loss_fn,optimizer):
    losses = []

    x = torch.from_numpy(x).view(-1,1).float()
    y = torch.from_numpy(y).view(-1,1).float()

    for epoch in range(num_epochs):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    return losses



def main():

    model1 = NeuralNetwork()
    model2 = NeuralNetwork()

    num_epochs = 1000
    learning_rate = 0.001

    func_type = "werid functions"
    xtrain = np.random.uniform(0, 1, 200)
    ytrain_s = np.array([cantor(x) for x in xtrain])
    ytrain_ns = np.array([weierstrass(x) for x in xtrain])

    loss_fn = nn.MSELoss()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate)

    convergence_s = train_loop(xtrain, ytrain_s, model1, num_epochs, learning_rate, loss_fn, optimizer1)
    convergence_ns = train_loop(xtrain, ytrain_ns, model2, num_epochs, learning_rate, loss_fn, optimizer2)

    plt.plot(convergence_s, label = "cantor")
    plt.plot(convergence_ns, label = "weierstrass")
    plt.legend()
    plt.title("Convergence of Neural Network for " + func_type)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"plots/convergence_{func_type}.png")
    plt.show()



if __name__ == '__main__':
    main()
