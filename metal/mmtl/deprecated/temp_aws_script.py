import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from metal.mmtl.metal_model import MetalModel
from metal.mmtl.task import Task
from metal.mmtl.trainer import MultitaskTrainer


def make_dataloader(n):
    X = np.random.random((n, 2)) * 2 - 1
    Y = (X[:, 0] > X[:, 1] + 0.25).astype(int) + 1

    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.long)

    div1 = int(n * 0.8)
    div2 = int(n * 0.9)
    Xs = [X[:div1], X[div1:div2], X[div2:]]
    Ys = [Y[:div1], Y[div1:div2], Y[div2:]]

    dataset = TensorDataset(Xs[0], Ys[0])
    data_loader = DataLoader(dataset, batch_size=4)
    return data_loader


foo_input = nn.Linear(2, 10)
bar_input = foo_input  # nn.Linear(100, 7)

foo_head = nn.Linear(10, 2)
bar_head = nn.Linear(10, 2)

foo_data = make_dataloader(5000)
bar_data = make_dataloader(2000)

foo = Task("foo_task", {"train": foo_data, "valid": foo_data}, foo_input, foo_head)
bar = Task("bar_task", {"train": bar_data, "valid": bar_data}, bar_input, bar_head)
# baz = Task("baz_task", "baz_head", [make_dataloader(100), None, None])
tasks = [foo, bar]


model = MetalModel(tasks, device=-1, verbose=False)
trainer = MultitaskTrainer()
trainer.train_model(
    model,
    tasks,
    n_epochs=3,
    lr=0.1,
    progress_bar=True,
    log_every=1,
    score_every=1,
    checkpoint_best=True,
    checkpoint_metric="foo_task/valid/accuracy",
    checkpoint_metric_mode="max",
)

for batch in foo.data_loaders["train"]:
    X, Y = batch
    print(model(X, ["foo_task"]))
    print(model.calculate_loss(X, Y, ["foo_task"]))
    print(model.calculate_probs(X, ["foo_task"]))
    break

print("SUCCESS!")
