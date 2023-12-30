import torch
import numpy as np
import numpy as np
import torch
import joblib
import tqdm


class MemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        now: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ):
        """
        This dataset allows the user
        to wrap another dataset and
        load all of the outputs into memory,
        so that they are accessed from RAM
        instead of storage. All attributes of
        the original dataset will still be available, except
        for :code:`._dataset` and :code:`._data_dict` if they
        were defined.
        It also allows the data to be saved in memory right
        away or after the data is accessed for the first time.


        Examples
        ---------

        .. code-block::

            >>> dataset = MemoryDataset(dataset, now=True)


        Arguments
        ---------

        - dataset: torch.utils.data.Dataset:
            The dataset to wrap and add to memory.

        - now: bool, optional:
            Whether to save the data to memory
            right away, or the first time the
            data is accessed. If :code:`True`, then
            this initialisation might take some time
            as it will need to load all of the data.
            Defaults to :code:`True`.

        - verbose: bool, optional:
            Whether to print progress
            as the data is being loaded into
            memory. This is ignored if :code:`now=False`.
            Defaults to :code:`True`.

        - n_jobs: int, optional:
            The number of parallel operations when loading
            the data to memory.
            Defaults to :code:`1`.


        """

        self._dataset = dataset
        self._data_dict = {}
        if now:
            pbar = tqdm.tqdm(
                total=len(dataset),
                desc="Loading into memory",
                disable=not verbose,
                smoothing=0,
            )

            def add_to_dict(index):
                for ni, i in enumerate(index):
                    self._data_dict[i] = dataset[i]
                    pbar.update(1)
                    pbar.refresh()
                return None

            all_index = np.arange(len(dataset))
            index_list = [all_index[i::n_jobs] for i in range(n_jobs)]

            joblib.Parallel(
                n_jobs=n_jobs,
                backend="threading",
            )(joblib.delayed(add_to_dict)(index) for index in index_list)

            pbar.close()

        return

    def __getitem__(self, index):
        if index in self._data_dict:
            return self._data_dict[index]
        else:
            output = self._dataset[index]
            self._data_dict[index] = output
            return output

    def __len__(self):
        return len(self._dataset)

    # defined since __getattr__ causes pickling problems
    def __getstate__(self):
        return vars(self)

    # defined since __getattr__ causes pickling problems
    def __setstate__(self, state):
        vars(self).update(state)

    def __getattr__(self, name):
        if hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        else:
            raise AttributeError
