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
    ''' This class is an alternative to the BaseDataset class

    Instead of loading all data once, this dataloader loads data batch by batch

    '''
    extensions = ["jpg", "jpeg", "png"]
    def __init__(self, root: str, file_list=None, output_size=(64, 64)):
        ''' Initialize the dataset with a directory path (and a potential file list)

        If the filelist is provided, then we simply concatenate the file listed in the list,

        Otherwise we scan through the directory to find all the files with desired extension

        Args:
            root (str): path to the directory
            file_list Union(str, list): path to the file_list (optional), or the actual list of filenames
            output_size (tuple(int, int)): due to the nature of custom datasets, 
                               its necessary to know the desired output size

        '''
        self.filenames = []
        self.output_size = output_size
        self.target_device = None
        self.shape = [1, 3, output_size[0], output_size[1]]
        if file_list: # if a filelist is provided
            if type(file_list) == str:
                lines = []
                with open(file_list, "r") as f:
                    lines = f.readlines()
            elif type(file_list) == list:
                lines = file_list
            else:
                raise ValueError("Cannot recognize the provided filelist for the dataset")
            if root:
                for line in lines:
                    self.filenames.append("{:s}/{:s}".format(root, line))
            else:
                self.filenames = lines
        else: # if the filelist is not provided, we scan through all files under the folder
            for ext in self.extensions:
                self.filenames.extend(glob.glob(f"{root}/**/*.{ext}", recursive = True))


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
            image = Image.open(filename)
            
            data = torch.Tensor(self.__resize__(asarray(image))).permute(2, 0, 1)
            if self.target_device is not None:
                data = data.to(self.target_device)
            if data.max() > 1.0:
                data = data/255
            if (data != data).sum() > 0: # if data has nan
                data = self.__getitem__((index + 1)%len(self.filenames))
            data_batch.append(data)
        if len(data_batch) == 1:
            return {"data": data_batch[0], "label": torch.zeros(1)}
        else:
            return {"data": torch.stack(data_batch), "label": torch.zeros(1)}

    def __resize__(self, image):
        '''Resize the image to the desired shape
        '''
        current_h, current_w, c = image.shape
        target_h, target_w = self.output_size
        target_ar = target_h / target_w

        h_needed = int(target_ar * current_w)
        if h_needed <= current_h: # if the current_height can help reach the target aspect ratio with change w
            w_needed = current_w
        else:
            w_needed = int(current_h / target_ar)
            h_needed = current_h
        
        image = image[(current_h-h_needed)//2:(current_h-h_needed)//2 + h_needed, (current_w-w_needed)//2:(current_w-w_needed)//2 + w_needed]
        image = resize(image, [target_h, target_h])
        return image

            





if __name__ == "__main__":
    dataset = FolderDataset("../../../../../data")
    print(dataset[0]['data'].size())