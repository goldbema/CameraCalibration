import numpy as np
import os

def load_dataset(dataset_dir_path):
    """Load data points and model points in a directory.
       Note that the model must be named 'model.txt', but all other
       image point sets are identified in lexicographic order.
    """
    dataset = {'data'  : [],
               'model' : []}

    for file in os.listdir(dataset_dir_path):
        path = os.path.join(dataset_dir_path, file)
        ext = os.path.splitext(file)[-1].lower()
        
        if ext == '.txt':
            data = load_data(path)

            key = 'data'
            if file.lower() == 'model.txt':
                key = 'model'

            dataset[key].append(data)

    return dataset

def load_data(data_path):
    """Load a single Nx2 point set.
    """
    dataset = []
    point = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            for item in line:
                point.append(float(item))
                if len(point) == 2:
                    dataset.append(point)
                    point = []
    if len(point) != 0:
        raise ValueError('Dataset contains odd number of entries')

    dataset = np.array(dataset)

    return dataset