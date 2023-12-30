import numpy as np
import tqdm
import copy
import torch
import torch.nn as nn
import typing as t


class EarlyStopper:
    def __init__(
        self,
        patience: int = 1,
        min_delta: int = 0,
        save_model: bool = False,
        model_path: t.Union[None, str] = None,
    ):
        """
        The early stopping class.
        This class is used to stop the training process if the validation
        loss does not improve for a certain number of epochs.



        Examples
        ---------

        .. code-block::

            >>> early_stopper = EarlyStopper(
            ...     patience=patience, min_delta=min_delta
            ... )
            >>> for epoch in range(n_epochs):
            ...     for data in train_loader:
            ...         # train the model
            ...     val_loss = evaluate(model, val_loader, device)
            ...     if early_stopper.early_stop(val_loss):
            ...         break


        Arguments
        ---------

        - patience: int, optional:
            The number of epochs to wait for the validation
            loss to improve before stopping the training.
            Defaults to :code:`1`.

        - min_delta: int, optional:
            The minimum amount of improvement in the validation
            loss to be considered an improvement.
            Defaults to :code:`0`.

        - save_model: bool, optional:
            Whether to save the model
            when the validation loss improves.
            Defaults to :code:`False`.

        - model_path: t.Union[None, str], optional:
            The path to save and load the model to and from.
            If this is :code:`None` then the model will be stored in memory.
            This is only used if :code:`save_model` is :code:`True`.
            Defaults to :code:`None`.


        """
        self.patience = patience if patience is not None else np.inf
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.save_model = save_model
        self.model_path = model_path

    def early_stop(
        self,
        validation_loss: float,
        model: t.Union[None, nn.Module] = None,
    ) -> bool:
        """
        Check whether training should be stoppped.


        Arguments
        ---------

        - validation_loss: float:
            The current validation loss.

        - model: t.Union[None, nn.Module], optional:
            The model to save if the validation loss improves.
            This is only used if :code:`self.save_model` is :code:`True`.
            Defaults to :code:`None`.


        Returns
        --------

        - out: bool:
            Whether to stop training or not.


        """
        if self.save_model:
            assert model is not None, "Model must be provided if save_model is True"

        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.save_model:
                if self.model_path is None:
                    self.state = copy.deepcopy(model.state_dict())
                else:
                    torch.save(model.state_dict(), f"{self.model_path}")

        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False

    def load_best_model(self, model: nn.Module) -> nn.Module:
        """
        Loads the best model if :code:`self.save_model` is :code:`True`.
        If :code:`self.save_model` is :code:`False` then this function
        will return the model unchanged.


        Arguments
        ---------

        - validation_loss: float:
            The current validation loss.

        - model: nn.Module:
            The model to update or return.


        Returns
        --------

        - model: nn.Module:
            The model with the best parameters loaded.


        """

        if self.save_model:
            if self.model_path is None:
                model.load_state_dict(self.state)
            else:
                model.load_state_dict(torch.load(f"{self.model_path}"))

        return model


# define training function that can be used
# with any model, loss function, data and optimiser
def train(
    model,
    train_loader,
    n_epochs,
    optimiser,
    criterion,
    val_loader=None,
    patience=5,
    scheduler=None,
):
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
        Defaults to :code:`None`.
    - patience: integer:
        The number of epochs to wait for the validation
        loss to improve before stopping the training.
        Defaults to :code:`5`.

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

    # ==== set up early stopper ====
    early_stopper = EarlyStopper(
        patience=patience,
        min_delta=1e-5,
        save_model=True,
    )

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
            # ==== update progress bar ====
            pbar.update(1)
            pbar.refresh()
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

    # loss stats
    train_loss_dict = {"step": [], "loss": []}
    val_loss_dict = {"step": [], "loss": []}

    tqdm.tqdm._instances.clear()

    # train for the given n_epochs
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(
            desc=f"Training on Epoch {epoch+1}/{n_epochs}", total=len(train_loader)
        )
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

        postfix = {"train_loss": avg_loss_train}


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
            # checking early stopper
            if early_stopper.early_stop(avg_loss_val, model):
                pbar.set_postfix(postfix)
                pbar.refresh()
                pbar.close()
                print("Stopping early...")
                break
            postfix["val_loss"] = avg_loss_val
        else:
            avg_loss_val = np.nan

        # ==== update scheduler ====
        if scheduler is not None:
            scheduler.step()
            postfix["lr"] = scheduler.get_last_lr()[0]

        # ==== set pbar info and update ====
        pbar.set_postfix(postfix)
        pbar.refresh()
        pbar.close()

    if val_loader is not None:
        print("Loading best model...")
        model = early_stopper.load_best_model(model)

    # put the model back on the cpu
    model.to("cpu")

    return model, (train_loss_dict, val_loss_dict)
