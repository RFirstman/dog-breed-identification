# Code (mostly) sourced from https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py
# This project is focused moreso on the experimentation and results, so I did not want to reinvent the wheel here
from PIL import Image
import os.path as path
import os
import scipy.io

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files
from sklearn.model_selection import train_test_split

from data.dog_classes import classes


class StanfordDogsDatasetHandler():
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        train (bool, optional): If true, get training data. If false, get test data.
        split (string, optional): Which split of data to construct ('train', 'val', or 'test')
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = "StanfordDogsData"
    url_prefix = "http://vision.stanford.edu/aditya86/ImageNetDogs/"

    def __init__(self,
                 root,        
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=True,
                 verbose=False):
        
        self.root = path.join(path.expanduser(root), self.folder)
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose

        self.classes = classes

        if download:
            self.download()

        self.split_train, self.split_val, self.split_test = self.load_data_split()


    def train_dataset(self):
        return StanfordDogsDataset(
            split = self.split_train,
            root = self.root,
            cropped = self.cropped,
            transform = self.transform,
            target_transform = self.target_transform,
            verbose = self.verbose
        )
    
    def val_dataset(self):
        return StanfordDogsDataset(
            split = self.split_val,
            root = self.root,
            cropped = self.cropped,
            transform = self.transform,
            target_transform = self.target_transform,
            verbose = self.verbose
        )
    
    def test_dataset(self):
        return StanfordDogsDataset(
            split = self.split_test,
            root = self.root,
            cropped = self.cropped,
            transform = self.transform,
            target_transform = self.target_transform,
            verbose = self.verbose
        )

    def download(self):
        import tarfile

        image_path = path.join(self.root, 'Images')
        annotation_path = path.join(self.root, 'Annotation')
        if path.exists( image_path ) and path.exists( annotation_path ):
            # There should be 120 directories for Images and Annotations (120 classes)
            if len(os.listdir(image_path)) == len(os.listdir(annotation_path)) == 120:
                print("Stanford Dogs Dataset already downloaded")
                return

        for filename in ['images.tar', 'annotation.tar', 'lists.tar']:
            file_url = self.url_prefix + filename
            download_url(file_url, self.root, filename=filename)

            if self.verbose:
                print("Extracting downloaded file: ", path.join(self.root, filename))
            
            with tarfile.open(path.join(self.root, filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            
            # clean up .tar file
            os.remove(path.join( self.root, filename ))

    def load_data_split(self):

        mat_file_train = scipy.io.loadmat(path.join(self.root, "train_list.mat"))
        mat_file_test = scipy.io.loadmat(path.join(self.root, "test_list.mat"))

        split_train = mat_file_train["annotation_list"]
        split_train = [ item[0][0] for item in split_train ]
        labels_train = mat_file_train["labels"]
        labels_train = [ item[0] - 1 for item in labels_train ]

        split_test = mat_file_test["annotation_list"]
        split_test = [ item[0][0] for item in split_test ]
        labels_test = mat_file_test["labels"]
        labels_test = [ item[0] - 1 for item in labels_test ]

        split_train, split_val, labels_train, labels_val = train_test_split(split_train, labels_train, test_size= 0.2, stratify=labels_train)

        return list(zip(split_train, labels_train)), list(zip(split_val, labels_val)), list(zip(split_test, labels_test))


class StanfordDogsDataset(data.Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        split (string, optional): The train, validation, or test split
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    folder = "StanfordDogsData"
    url_prefix = "http://vision.stanford.edu/aditya86/ImageNetDogs/"

    def __init__(self,
                 split,
                 root,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 verbose=False):
        
        # self.root = path.join(path.expanduser(root), self.folder)
        self.root = root
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose

        self.classes = classes

        self.images_folder = path.join(self.root, 'Images')
        self.annotation_folder = path.join(self.root, 'Annotation')
        self._dog_breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, bounding_box, index)
                                        for bounding_box in self.get_bounding_boxes(path.join(self.annotation_folder, annotation))]
                                        for annotation, index in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation+'.jpg', index) for annotation, _, index in self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation+'.jpg', index) for annotation, index in split]

            self._flat_breed_images = self._breed_images

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class


    @staticmethod
    def get_bounding_boxes(annotation_path):
        import xml.etree.ElementTree
        element = xml.etree.ElementTree.parse(annotation_path).getroot()
        boxes = []
        for objs in element.iter('object'):
            x_min = int(objs.find('bndbox').find('xmin').text)
            y_min = int(objs.find('bndbox').find('ymin').text)
            x_max = int(objs.find('bndbox').find('xmax').text)
            y_max = int(objs.find('bndbox').find('ymax').text)
            boxes.append([x_min, y_min, x_max, y_max])
        return boxes

    def get_stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)"%(len(self._flat_breed_images), len(counts.keys()), float(len(self._flat_breed_images))/float(len(counts.keys()))))

        return counts
