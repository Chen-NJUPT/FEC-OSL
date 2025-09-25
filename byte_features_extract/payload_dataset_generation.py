__author__ = 'HPC'
import json
import numpy as np
import os
import torch
from torch.utils.data import random_split
import os
from torchvision import datasets, transforms
import math

def build_dataset(args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(args.data_path)
    items = os.listdir(args.data_path)
    nb_classes = len(items)
    dataset = datasets.ImageFolder(root, transform=transform)
    total_size = len(dataset)
    nb_classes_known = int(nb_classes * args.ratio)
    nb_classes_unknown = math.ceil((nb_classes - nb_classes_known) / 2)
    nb_classes_aunknown = math.floor((nb_classes - nb_classes_known) / 2)
    
    index_known = (0, nb_classes_known * int(len(dataset) / nb_classes))
    index_aunknown = (nb_classes_known * int(len(dataset) / nb_classes), 
                     (nb_classes_known + nb_classes_aunknown) * int(len(dataset) / nb_classes))
    index_unknown = ((nb_classes_known + nb_classes_aunknown) * int(len(dataset) / nb_classes), total_size)
    
    known_dataset = torch.utils.data.Subset(dataset, list(range(*index_known)))
    aunknown_dataset = torch.utils.data.Subset(dataset, list(range(*index_aunknown)))
    unknown_dataset = torch.utils.data.Subset(dataset, list(range(*index_unknown)))
    
    # known
    train_size = int(0.8 * len(known_dataset))
    val_size = len(known_dataset) - train_size 
    train_dataset, val_dataset = random_split(known_dataset, [train_size, val_size])
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    # aunknown
    train_size2 = int(0.8 * len(aunknown_dataset))
    val_size2 = len(aunknown_dataset) - train_size2
    aunknown_train_dataset, aunknown_val_dataset  = random_split(aunknown_dataset, [train_size2, val_size2])
    aunknown_train_indices = aunknown_train_dataset.indices
    aunknown_val_indices = aunknown_val_dataset.indices
    
    # unknown
    train_size3 = int(0.8 * len(unknown_dataset))
    val_size3 = len(unknown_dataset) - train_size3
    unknown_train_dataset, unknown_val_dataset  = random_split(unknown_dataset, [train_size3, val_size3])
    unknown_train_indices = unknown_train_dataset.indices
    unknown_val_indices = unknown_val_dataset.indices
    
    dic = {}
    dic["known_train_indices"] = train_indices
    dic["known_val_indices"] = val_indices
    dic["aunknown_train_indices"] = [(i + len(known_dataset)) for i in aunknown_train_indices]
    
    offset = len(known_dataset) + len(aunknown_dataset)

    unknown1_train = [(i + len(known_dataset)) for i in aunknown_train_indices]  # aunknown
    unknown1_val = [(i + len(known_dataset)) for i in aunknown_val_indices]
    
    unknown2_train = [(i + offset) for i in unknown_train_indices] # unknown
    unknown2_val = [(i + offset) for i in unknown_val_indices]
    
    # aunknown + unknown
    combined_train = unknown1_train + unknown2_train
    np.random.shuffle(combined_train)
    combined_val = unknown1_val + unknown2_val
    np.random.shuffle(combined_val)
    dic["unknown_val_indices"] = combined_val

    with open('dataset/byte_data/data_index_' + str(nb_classes_known) + "_" + str(nb_classes - nb_classes_known) +  '.json', 'w') as json_file:
        json.dump(dic, json_file, indent=4)
        
    with open('dataset/byte_data/data_index_' + str(nb_classes_known) + "_" + str(nb_classes - nb_classes_known) +  '.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        train_dataset_known = [dataset[i] for i in data["known_train_indices"]]
        val_dataset_known = [dataset[i] for i in data["known_val_indices"]]
        
        train_dataset_aunknown = [dataset[i] for i in data["aunknown_train_indices"]] # aunknown
        val_dataset_unknown = [dataset[i] for i in data["unknown_val_indices"]] # aunknown + unknown

    """
        train_dataset_known: Training split containing only known classes
        val_dataset_known: Validation split over the same known label space as training 
        train_dataset_aunknown: Auxiliary unknown training data without fine-grained labels; map all samples to a single 'unknown' label 
        val_dataset_unknown: Validation set for unknowns, covering both auxiliary-unknown and entirely unseen categories
    """
    return train_dataset_known, val_dataset_known, train_dataset_aunknown, val_dataset_unknown