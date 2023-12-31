import typing
import tqdm
import torch
import torch.nn as nn
import numpy as np


# define training function that can be used
# with any model, loss function, data and optimiser
def train(model, train_loader, n_epochs, optimiser, criterion, val_loader=None):
    """
    A function to train any model with a given dataset, optimiser, and
    criterion (loss function).

    Arguments
    ---------

    - model: pytorch nn object:
        The model to train
    - train_loader: pytorch data loader:
        The data to train with.
    - n_epochs: integer:
        The number of epochs to train for.
    - optimiser: pytorch optimiser:
        The optimiser to make the model updates.
    - criterion: pytorch nn object:
        The loss function to calculate the loss with.
    - val_loader: pytorch data loader:
        The data to calculate the validation loss with.

    Returns
    ---------

    - model: pytorch nn object:
        Trained version of the model given
        as an input.
    - tuple of dictionaries:
        - train_loss_dict: dictionary:
            Dictionary containing the training loss
            with keys: `steps` and `loss`.
        - val_loss_dict
            Dictionary containing the validation loss
            with keys: `steps` and `loss`.

    """
    # check if GPU is available and use that if so
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ==== push model to GPU if available ====
    model.to(device)

    # note that since all the following functions in this
    # function rely on the inputs above,
    # they won't work outside of the train function

    # pass a single batch of data through the model and get loss
    def batch_loss(inputs, targets):
        # ==== push data to GPU if available ====
        inputs, targets = inputs.to(device), targets.to(device)
        # ==== forward pass ====
        outputs = model(inputs)
        # ==== calc and save loss ====
        loss = criterion(outputs, targets)
        # ==== return loss ====
        return loss

    # train for an epoch
    def train_epoch(train_loader):
        model.train()  # set model option to train - important if using dropout or batchnorm
        batch_loss_list = []  # we will store all losses in a list
        # for each batch in the train loader
        for nb, (inputs, targets) in enumerate(train_loader):
            # ==== set gradient to zero ====
            optimiser.zero_grad()  # really important! Common mistake to not do this!
            # run data through batch_loss function to get loss
            loss = batch_loss(inputs=inputs, targets=targets)
            # ==== calc backprop gradients ====
            loss.backward()
            # ==== perform update step ====
            optimiser.step()
            # ==== store loss for later ====
            batch_loss_list.append(loss.item())
        # ==== return loss over batch ====
        return batch_loss_list

    # perform an epoch over the validation data to get loss
    @torch.no_grad()  # dont want gradients in validation since we're not training
    def val_epoch(val_loader):
        model.eval()  # set model option to eval - important if using dropout or batchnorm
        batch_loss_list = []  # we will store all losses in a list
        # for each batch in the val loader
        for nb, (inputs, targets) in enumerate(val_loader):
            # ==== set gradient to zero ====
            optimiser.zero_grad()  # gradients shouldnt be calculated but good practise
            # run data through batch to get loss
            loss = batch_loss(inputs=inputs, targets=targets)
            # ==== store loss for later ====
            batch_loss_list.append(loss.item())
        # ==== calculate average loss ====
        return batch_loss_list

    pbar = tqdm.tqdm(desc="Training", total=n_epochs)  # progress bar
    # loss stats
    train_loss_dict = {"step": [], "loss": []}
    val_loss_dict = {"step": [], "loss": []}

    # train for the given n_epochs
    for epoch in range(n_epochs):
        # ==== train for an epoch ====
        n_batches = len(train_loader)
        batch_lost_list_train = train_epoch(train_loader=train_loader)
        # ==== get loss stats ====
        train_loss_dict["loss"].extend(batch_lost_list_train)  # adding loss
        # adding step values. These are the number of steps from the beginning
        train_loss_dict["step"].extend(
            list(np.arange(epoch * n_batches, (epoch + 1) * n_batches) + 1)
        )
        avg_loss_train = np.mean(batch_lost_list_train)

        # if a validation loader is passed
        if val_loader is not None:
            # ==== epoch over validation ====
            batch_lost_list_val = val_epoch(val_loader=val_loader)
            # ==== get loss stats ====
            avg_loss_val = np.mean(batch_lost_list_val)
            val_loss_dict["loss"].append(avg_loss_val)
            val_loss_dict["step"].append(
                (epoch + 1) * n_batches
                + 1  # the number of new steps is as many as the train loader
            )
        else:
            avg_loss_val = np.nan

        # ==== set pbar info and update ====
        pbar.set_postfix(
            {"Train Loss": f"{avg_loss_train:.3f}", "Val Loss": f"{avg_loss_val:.3f}"}
        )
        pbar.update(1)
        pbar.refresh()

    # put the model back on the cpu
    model.to("cpu")

    return model, (train_loss_dict, val_loss_dict)
