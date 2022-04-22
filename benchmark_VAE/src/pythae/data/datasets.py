"""The pythae's Datasets inherit from
:class:`torch.utils.data.Dataset` and must be used to convert the data before
training. As of today, it only contains the :class:`pythae.data.BaseDatset` useful to train a
VAE model but other Datatsets will be added as models are added.
"""
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from numpy import asarray
from skimage.transform import resize

# import to load image with torchvision
from torchvision.io import read_image, write_jpeg
from torchvision import transforms
import numpy as np


class BaseDataset(Dataset):
    """This class is the Base class for pythae's dataset

    A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data`, `labels` etc...
    This Class should be used for any new data sets.
    """

    def __init__(self, data, labels, binarize=False):

        self.labels = labels.type(torch.float)

        if binarize:
            self.data = (torch.rand_like(digits) < data).type(torch.float)

        else:
            self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
            """
        # Select sample
        X = self.data[index]

        # Load data and get label
        # X = torch.load('data/' + DATA + '.pt')
        y = self.labels[index]

        return {"data": X, "labels": y}


class FolderDataset(Dataset):
    """ This class is an alternative to the BaseDataset class

    Instead of loading all data once, this dataloader loads data batch by batch

    """

    extensions = ["jpg", "jpeg", "png"]

    def __init__(self, root: str, file_list=None, output_size=(64, 64)):
        """ Initialize the dataset with a directory path (and a potential file list)

        If the filelist is provided, then we simply concatenate the file listed in the list,

        Otherwise we scan through the directory to find all the files with desired extension

        Args:
            root (str): path to the directory
            file_list Union(str, list): path to the file_list (optional), or the actual list of filenames
            output_size (tuple(int, int)): due to the nature of custom datasets, 
                               its necessary to know the desired output size

        """
        self.filenames = []
        self.output_size = output_size
        self.target_device = None
        self.shape = [1, 3, output_size[0], output_size[1]]
        if file_list:  # if a filelist is provided
            if type(file_list) == str:
                lines = []
                with open(file_list, "r") as f:
                    lines = f.readlines()
            elif type(file_list) == list:
                lines = file_list
            else:
                raise ValueError(
                    "Cannot recognize the provided filelist for the dataset"
                )
            if root:
                for line in lines:
                    self.filenames.append("{:s}/{:s}".format(root, line))
            else:
                self.filenames = lines
        else:  # if the filelist is not provided, we scan through all files under the folder
            for ext in self.extensions:
                self.filenames.extend(glob.glob(f"{root}/**/*.{ext}", recursive=True))

    def set_device(self, device):
        self.target_device = device

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
            """
        # Select sample

        filenames = self.filenames[index]
        if type(filenames) is str:
            filenames = [filenames]

        data_batch = []
        for filename in filenames:
            data = read_image(filename)

            data_batch.append(data)

        data = torch.stack(data_batch)

        if data.shape[-1] in [1, 3]:
            data = data.permute(0, 3, 1, 2)

        data = self.__resize__(data)

        data = data / 255

        data = data.squeeze(dim=0)

        return {"data": data, "label": torch.zeros(1)}

    def __resize__(self, image):
        """Resize the image to the desired shape
        """

        _, c, current_h, current_w = image.shape

        target_h, target_w = self.output_size

        # get smaller side
        smaller_side = min(current_h, current_w)

        # center crop
        resizer = transforms.Compose(
            [
                transforms.CenterCrop((smaller_side, smaller_side)),
                transforms.Resize((target_h, target_w)),
            ]
        )

        image = resizer(image)

        return image


if __name__ == "__main__":
    dataset = FolderDataset("../../../../../data")
    print(dataset[0]["data"].size())

