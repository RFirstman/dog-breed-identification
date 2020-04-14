# Code (mostly) sourced from https://github.com/zrsmithson/Stanford-dogs/blob/master/data/load.py
# This project is focused moreso on the experimentation and results, so I did not want to reinvent the wheel here

from torchvision import transforms, datasets
from data.stanford_dogs import StanfordDogsDatasetHandler, StanfordDogsDataset

def load_datasets(root, input_size=224, print_stats=True):
    input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    
    dataset_handler = StanfordDogsDatasetHandler(
        root,
        cropped=False,
        transform=input_transforms,
        download=True,
        verbose=False,
    )

    train = dataset_handler.train_dataset()
    val = dataset_handler.val_dataset()
    test = dataset_handler.test_dataset()

    # train = StanfordDogsDataset(
    #             root,
    #             train=True,
    #             cropped=False,
    #             transform=input_transforms,
    #             download=True
    #         )

    # test = StanfordDogsDataset(
    #             root,
    #             train=False,
    #             cropped=False,
    #             transform=input_transforms,
    #             download=True
    #         )
    
    classes = train.classes

    if print_stats:
        print("Training set stats:")
        train.get_stats()
        print("Validation set stats:")
        val.get_stats()
        print("Testing set stats:")
        test.get_stats()

    return train, val, test, classes