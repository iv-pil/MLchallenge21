"""
Author: Ivan Pilkov
Matr.Nr.: K12049126
Exercise 5

datasets.py

"""

from torchvision import transforms
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import dill as pkl


def resize_imgs(inp_dir, out_dir, im_shape=90):
    """
    Function that recursively resizes jpg imgs in a folder
    //Used once in preprocess of whole dataset for ease of computation
    :param inp_dir: source folder
    :param out_dir: output folder; set out_dir=inp_dir for in-place processing
    :param im_shape: desired image size in pixels (default=90x90)
    :return: number of succesfully processed images
    """
    output = os.path.abspath(out_dir)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    # Scan inp_dir and sort files
    files = glob.glob(os.path.join(inp_dir, '**'), recursive=True)
    files.sort(key=str)
    files.pop(0)
    # Process sorted files and return number of transformed jpg imgs
    resize_transforms = transforms.Compose([
        transforms.Resize(size=im_shape),
        transforms.CenterCrop(size=(im_shape, im_shape)),
    ])
    n_files = 0
    for path in files:
        if path.lower().endswith((".jpg", ".jpeg")):
            image = Image.open(os.path.join(inp_dir, path))
            image = resize_transforms(image)
            output = os.path.join(out_dir, os.path.relpath(path, start=inp_dir))
            outpdir = os.path.dirname(os.path.abspath(output))
            if not os.path.exists(outpdir):
                os.makedirs(outpdir, exist_ok=True)
            image.save(output)
            n_files += 1
    return n_files


def datareader(image_array, border_x: (int, int), border_y: (int, int)):
    """
    A function that takes an image(ndarray) as input and returns the inputs
    and targets for training
    :param image_array: ndarray type
    :param border_x: tuple of pixel values
    :param border_y: tuple of pixel values
    :return: input_array, known_array, target_array
    """
    # Check type of image_array
    if not isinstance(image_array, np.ndarray):
        raise NotImplementedError
    if np.ndim(image_array) != 2:
        raise NotImplementedError
    # Check border_x and border_y datatype
    x = list(border_x)
    y = list(border_y)
    x[0] = int(x[0])
    x[1] = int(x[1])
    y[0] = int(y[0])
    y[1] = int(y[1])
    # Check border values
    if x[0] < 1 or x[1] < 1:
        raise ValueError
    if y[0] < 1 or y[1] < 1:
        raise ValueError
    # Calculate the shape of the remaining known image pixels
    # and check if it raises error
    remain = np.subtract(np.shape(image_array), (x[0]+x[1], y[0]+y[1]))
    if remain[0] < 16 or remain[1] < 16:
        raise ValueError
    # Create known_array
    known_array = np.ones_like(image_array)
    for i in range(x[0]): known_array[i, :] = 0
    for j in range(x[1]): known_array[-(j+1), :] = 0
    for i in range(y[0]): known_array[:, i] = 0
    for j in range(y[1]): known_array[:, -(j+1)] = 0
    # Create target_array by applying known_array as a bool mask on image_array
    target_array = image_array[known_array == 0]
    # Create input_array
    input_array = np.where(known_array == 1, image_array, 0)
    return input_array, known_array, target_array


def norm(array):
    array = np.array(array, dtype=np.float32)
    min = np.min(array)
    array -= min
    max = np.max(array)
    array /= max
    array *= 2
    array -= 1
    return array, min, max


def denorm(array, min, max):
    return (array + 1) / 2 * max + min


class Dataset5(Dataset):
    def __init__(self, root):
        """
        For now - loads the whole dataset as raw ndarray,
        all images stacked at 0 dimension

        ?Optional - save data_array to a pickle file

        :param root: image folder
        """
        img_paths = glob.glob(os.path.join(root, '**/*.jpg'), recursive=True)
        img_paths.sort(key=str)
        data_array = np.empty(shape=(len(img_paths), 90, 90), dtype=np.uint8)
        for i, n in enumerate(img_paths):
            img = Image.open(n)
            data_array[i, ...] = img
        self.data = data_array
        self.root = root

    def __getitem__(self, idx):
        """
        Should take a single image, create random borders, pass it through datareader
        then normalize it, return stacked tensor of input+known_arrays, target, id and norm consts
        :param idx:
        :return:
        """
        img = self.data[idx, ...]
        # To include (optional) normalization

        # only for debugging reproducibility, remove when training
        np.random.seed(idx)

        x0 = np.random.randint(5, 11)
        x1 = np.random.randint(5, 16-x0)
        y0 = np.random.randint(5, 11)
        y1 = np.random.randint(5, 16 - y0)
        max_len_tar = 90*90 - 75*75

        #img, min, max = norm(img)

        input, known, _target = datareader(img, (x0, x1), (y0, y1))
        #known, _, _ = norm(known)
        stacked_in = np.stack((input, known), axis=0).astype(np.float32)
        #stacked_in, min, max = norm(stacked_in)
        _target = np.asarray(_target, dtype=np.float32)
        target = np.zeros(max_len_tar, dtype=np.float32)
        target[:len(_target)] = _target
        id = idx

        #return stacked_in, target, id, min, max
        return stacked_in, target, id

    def __len__(self):
        return len(self.data)


class DataTest(Dataset5):
    def __init__(self, root):
        super().__init__(root)

    def __getitem__(self, item):
        img = self.data[item, ...]
        np.random.seed(item)
        x0 = np.random.randint(5, 11)
        x1 = np.random.randint(5, 16 - x0)
        y0 = np.random.randint(5, 11)
        y1 = np.random.randint(5, 16 - y0)
        input, known, target = datareader(img, (x0, x1), (y0, y1))
        return input, known, target, item


def create_test(source, output_test, output_targ):
    test = DataTest(source)
    input_arrays = []
    known_arrays = []
    sample_ids = []
    targets = []
    for i in range(len(test)):
        input, known, target, item = test[i]
        input_arrays.append(input)
        known_arrays.append(known)
        sample_ids.append(item)
        targets.append(target)

    tset = {"input_arrays": input_arrays,
            "known_arrays": known_arrays,
            "sample_ids": sample_ids}

    with open(output_test, "wb") as pk1:
        pkl.dump(tset, pk1)
        pk1.close()

    with open(output_targ, "wb") as pk2:
        pkl.dump(targets, pk2)
        pk2.close()
