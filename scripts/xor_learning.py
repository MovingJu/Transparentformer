"""Training and evaluating xor classifier"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transparentformer.models.xor import XorClassifier, XorClassifierLinear

OUTPUT_DIR = Path(__file__).resolve().parents[0] / "outputs"


def main():
    # section: Non-linear model
    # section: definition
    train = torch.Tensor(((0, 0), (0, 1), (1, 0), (1, 1)))
    target = torch.Tensor(((0,), (1,), (1,), (0,)))
    assert train.shape == (4, 2)
    assert target.shape == (4, 1)
    epochs = 500
    
    # section: training
    model = XorClassifier()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
    loss_log_nonlin: list[np.ndarray] = []
    for epoch in range(epochs):
        pred = model(train)
        loss: torch.Tensor = loss_fn(pred, target)
        loss_log_nonlin.append(loss.detach().numpy())
        loss.backward() # pyright: ignore
        optimizer.step()  # pyright: ignore
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"[{epoch:>4}] loss: {loss:.4f}")

    # section: testing
    with torch.no_grad():
        model.eval()
        for point in train:
            print(f"[point] ({point[0]}, {point[1]}) : {model(point).round()}")

    # section: drawing
    cartesian_2 = torch.Tensor(
        [(i, j) for i in np.linspace(0, 1, 100) for j in np.linspace(0, 1, 100)]
    )
    fig, ax = plt.subplots()
    ax.scatter(  # pyright: ignore
        cartesian_2.detach().numpy()[:, 0],
        cartesian_2.detach().numpy()[:, 1],
        c=model(cartesian_2).detach().numpy(),
        cmap="coolwarm",
    )
    fig.savefig(OUTPUT_DIR / "xor_nonlin.png")  # pyright: ignore
    plt.close(fig)

    # section: linear model
    model = XorClassifierLinear()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-2)
    loss_log_lin: list[np.ndarray] = []
    for epoch in range(epochs):
        pred = model(train)
        loss = loss_fn(pred, target)
        loss_log_lin.append(loss.detach().numpy())
        loss.backward() # pyright: ignore
        optimizer.step()  # pyright: ignore
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"[{epoch:>4}] loss: {loss:.4f}")

    # section: testing
    with torch.no_grad():
        model.eval()
        for point in train:
            print(f"[point] ({point[0]}, {point[1]}) : {model(point).round()}")

    # section: drawing
    cartesian_2 = torch.Tensor(
        [(i, j) for i in np.linspace(0, 1, 100) for j in np.linspace(0, 1, 100)]
    )
    fig, ax = plt.subplots()
    ax.scatter(  # pyright: ignore
        cartesian_2.detach().numpy()[:, 0],
        cartesian_2.detach().numpy()[:, 1],
        c=model(cartesian_2).detach().numpy(),
        cmap="coolwarm",
    )
    fig.savefig(OUTPUT_DIR / "xor_lin.png")  # pyright: ignore
    plt.close(fig)

    fig, ax = plt.subplots()
    # section: evaluating
    ax.plot([i for i in range(epochs)], loss_log_nonlin, label="Non-linear") # pyright: ignore
    ax.plot([i for i in range(epochs)], loss_log_lin, label="Linear") # pyright: ignore
    ax.set_xlabel("epochs") # pyright: ignore
    ax.set_ylabel("loss") # pyright: ignore
    ax.legend() # pyright: ignore
    fig.savefig(OUTPUT_DIR / "xor_log.png")  # pyright: ignore
    plt.close(fig)

if __name__ == "__main__":
    main()
