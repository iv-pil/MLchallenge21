"""
Author: Ivan Pilkov
Matr.Nr.: K12049126
Exercise 5

main.py

"""

import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import tqdm

from datasets import Dataset5
from datasets import datareader, norm, denorm
from architectures import SimpleCNN


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, file_names = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs for network
            outputs = model(inputs)
            transformed_outputs = torch.zeros((len(outputs), 2475), dtype=torch.float32)
            for i, sample in enumerate(outputs):
                sample = sample[0, inputs[i, 1] == 0]
                transformed_outputs[i, :len(sample)] = torch.unsqueeze(sample, 0)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
            # No need - norm/denorm -> values only within limits

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(transformed_outputs, target) for transformed_outputs, target
                                  in zip(transformed_outputs, targets)]).sum() / len(dataloader.dataset))
    return loss


def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), device: torch.device = torch.device("cpu")):
    """Main function that takes hyperparameters and performs training and evaluation of model
        Mostly copied from example project - plotting/removed, evaluating, writing to tensorboard, early stopping
    """

    # Load dataset
    root_dir = r"D:\MLchallenge21\dataset\resized"
    dataset = Dataset5(root=root_dir)
    print(dataset.root)
    print(len(dataset))

    # Split dataset into training, validation, and test set randomly
    trainingset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 4))))
    validationset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (6 / 8)),
                                                                        int(len(dataset) * ( 7/ 8))))
    testset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (7 / 8)),
                                                                        len(dataset)))
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=16, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 5e1  # print status to tensorboard every x updates
    validate_at = 1e3  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    # Train until n_updates update have been reached
    while update < n_updates:
        for data in trainloader:
            # Get next samples
            #inputs, targets, ids, min, max = data
            inputs, targets, ids = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = net(inputs)

            # Denormalize and apply known_array as a mask to get 1D target
            # For now using for loop - look into numpy operations for speedup,
            # maybe redeclare better norm function?
            transformed_outputs = torch.zeros((len(outputs), 2475), dtype=torch.float32)
            for i, sample in enumerate(outputs):
                #sample = denorm(sample, min[i], max[i])
                #mask = denorm(inputs[i, 1], 0, 1).to(dtype=bool)
                #sample = sample[0, ~mask]
                sample = sample[0, inputs[i, 1] == 0]
                transformed_outputs[i, :len(sample)] = torch.unsqueeze(sample, 0)

            # Calculate loss, do backward pass, and update weights
            loss = mse(transformed_outputs, targets)
            loss.backward()
            optimizer.step()

            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=valloader, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

    update_progess_bar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)


import dill as pkl


def net_pred(modelpath: str, datapath: str, outputpath: str):
    with open(datapath, "rb") as pf:
        exset = pkl.load(pf)
        pf.close()

    net = torch.load(modelpath)
    inputs = list(exset["input_arrays"])
    known = list(exset["known_arrays"])

    pred = []

    with torch.no_grad():
        for (sample, mask) in zip(inputs, known):
            #sample, min, max = norm(samp le)
            #mask, _, _ = norm(mask)
            inp = torch.from_numpy(sample).to(dtype=torch.float32)
            kno = torch.from_numpy(mask).to(dtype=torch.float32)
            stacked = torch.stack((inp, kno), dim=0)
            stacked = stacked.unsqueeze(dim=0)
            output = net(stacked)
            #mask = denorm(mask, 0, 1).astype(bool)

            mask = np.expand_dims(mask, axis=0).astype(bool)
            flat = output[0, ~mask]
            flat_array = flat.numpy()
            #flat_array = denorm(flat_array, min, max)
            flat_array = np.around(flat_array)
            flat_array = np.clip(flat_array, 0., 255.)
            flat_array = np.array(flat_array, dtype=np.uint8)
            pred.append(flat_array)

    with open(outputpath, "wb") as pkf:
        pkl.dump(pred, pkf)
        pkf.close()


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    main(**config)
