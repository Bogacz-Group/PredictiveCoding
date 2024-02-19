"""
This script examines a 'normal', or generative predictive coding model, with a supervised top-layer on its performance on the MNIST dataset.

Here, normal PCN, or generative PCN refers to the Rao and Ballard (1999) formulation of predictive coding, where the class labels are provided to the
    top layer to generate the images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random

import predictive_coding as pc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using {device}')

class Options:
    pass
options = Options()

options.batch_size = 500
options.train_size = 10000
options.test_size = 1000
options.layer_sizes = [10, 256, 256, 784]
options.activation = nn.Tanh()

def get_mnist(options):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Lambda(lambda x: torch.flatten(x))])
    train = datasets.MNIST('./data', train=True, transform=transform, download=True)
    test = datasets.MNIST('./data', train=False, transform=transform, download=True)
    
    if options.train_size != len(train):
        train = torch.utils.data.Subset(train, random.sample(range(len(train)), options.train_size))
    if options.test_size != len(test):
        test = torch.utils.data.Subset(test, random.sample(range(len(test)), options.test_size))

    # Split the training set into training and validation sets
    train_len = int(len(train) * 0.9)  # 80% of data for training
    val_len = len(train) - train_len   # remaining 20% for validation
    train_set, val_set = torch.utils.data.random_split(train, [train_len, val_len])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=options.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=options.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=options.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# top-layer prior
# when doing conditional generation, we can play with this
class BiasLayer(nn.Module):
    def __init__(self, num_features, offset=0.):
        super().__init__()
        self.bias = nn.Parameter(offset * torch.ones(num_features))

    def forward(self, x):
        return torch.zeros_like(x) + self.bias


"""Intialization functions for the top layer; the default initialization will set x to the vector of 0.1s - 84%
you can choose the following top-layer initializations to see their effects on the performance
"""

# initialization with a normal distribution centered at mu - 75%
def sample_x_fn_normal_mu(inputs):
    return torch.normal(inputs['mu'])

# using normal initialization harms performance - 72%
def sample_x_fn_normal(inputs):
    return torch.randn_like(inputs['mu'])

# doesn't work better than default, but improves simple normal initialization to 83%
def sample_x_fn_softmax(inputs):
    return F.softmax(torch.randn_like(inputs['mu']), dim=1)


# energy function for the top layer with a sparse L1 penalty
def sparse_l2_energy_fn(inputs, penalty=1e-2):
    return 0.5 * (inputs['mu'] - inputs['x'])**2 + penalty * torch.abs(inputs['x'])

def loss_fn(output, _target):
    return (output - _target).pow(2).sum() * 0.5

# test function
def test(model, loader):
    test_model = nn.Sequential(
        BiasLayer(options.layer_sizes[0], offset=0.1),
        pc.PCLayer(
            energy_fn=sparse_l2_energy_fn,
            energy_fn_kwargs={'penalty': 0.0},
        ),
        model
    ).to(device)

    test_model.train()

    pc_trainer = pc.PCTrainer(test_model, 
        T=100,
        update_x_at='all',
        optimizer_x_fn=optim.SGD,
        optimizer_x_kwargs={"lr": 0.1},
        update_p_at='never', # do not update parameters during inference
        plot_progress_at=[],
        x_lr_discount=0.5,
    )
    
    correct_cnt = 0
    for data, target in loader:
        data = data.to(device)
        target = F.one_hot(target, num_classes=10).to(torch.float32).to(device)
        results = pc_trainer.train_on_batch(
            inputs=target,
            loss_fn=loss_fn,
            loss_fn_kwargs={'_target': data},
            is_log_progress=False,
            is_return_results_every_t=False,
        )
        pred = test_model[1].get_x()
        correct = pred.argmax(dim=1).eq(target.argmax(dim=1)).sum().item()
        correct_cnt += correct
    return correct_cnt / len(loader.dataset)

model = nn.Sequential(
    nn.Linear(options.layer_sizes[0], options.layer_sizes[1]), # generates the prediction
    pc.PCLayer(),
    options.activation,
    nn.Linear(options.layer_sizes[1], options.layer_sizes[2]),
    pc.PCLayer(),
    options.activation,
    nn.Linear(options.layer_sizes[2], options.layer_sizes[3]),
).to(device)

# sample a batch of data
train_loader, val_loader, test_loader = get_mnist(options)

model.train()

pc_trainer = pc.PCTrainer(
    model,
    T=20,
    update_x_at='all',
    optimizer_x_fn=optim.SGD,
    optimizer_x_kwargs={"lr": 0.1},
    update_p_at='last',
    optimizer_p_fn=optim.Adam,
    optimizer_p_kwargs={"lr": 0.001, "weight_decay":0.001, "betas":(0.9,0.999)},
    plot_progress_at=[],
    x_lr_discount=0.5,
)

for i in range(10):
    print(f'epoch {i}')
    correct_cnt = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = F.one_hot(target, num_classes=10).to(torch.float32).to(device)
        # this will log the loss of each BATCH
        results = pc_trainer.train_on_batch(
            inputs=target,
            loss_fn=loss_fn,
            loss_fn_kwargs={
                '_target': data,
            },
            is_log_progress=False,
            is_return_results_every_t=False,
        )

    print(f'training accuracy: {test(model, train_loader)}')
    print(f'validation accuracy: {test(model, val_loader)}')

print(f'test accuracy: {test(model, test_loader)}')
# when testing/validation, set 'update_p_at' to 'never' (for genetaive direction)
