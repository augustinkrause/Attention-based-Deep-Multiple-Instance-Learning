import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
from tqdm import tqdm

TMP_PATH = "/tmp"
TASK_ID = int(os.environ["SLURM_JOB_ID"])
def get_bag_shape(bag_path):
    return (len(np.load(bag_path).files), 768)

def get_bag(bag_path : str):
    bag_file = np.load(bag_path)
    bag = np.zeros((len(bag_file.files), 768))
    for i in tqdm(range(len(bag_file.files)), desc=f"Reading slide."):
        patch = bag_file.files[i]
        bag[i] = bag_file[patch][0]
        
    return bag

class TCGADataset(Dataset):
    def __init__(self, n_train=None, n_test=None, n_bags=None, train = True, transformation=None, seed=1):
        
        self.transformation = transformation
        self.seed = seed
        np.random.seed(self.seed) # sets the random seed ALSO for all pandas functions

        # load metadata (for labels)
        self.case_metadata = pd.read_csv(os.path.join(os.getcwd(), "data", "datasets", "TCGA", "case_metadata.csv"))
        self.slide_metadata = pd.read_csv(os.path.join(os.getcwd(), "data", "datasets", "TCGA", "slide_metadata.csv"))
        cols = ["case_id", "slide_id", "ajcc_pathologic_t"]
        self.metadata = pd.merge(self.case_metadata, self.slide_metadata, how="inner")[cols]
        self.metadata.drop_duplicates(subset=["case_id"], keep="first", inplace=True) # only take one slide per case
        
        # store slides paths
        self.slide_paths = {slide_id : os.path.join(os.getcwd(), "data", "datasets", "TCGA", f"{slide_id}.npz") for slide_id in self.metadata["slide_id"]}
        self.n_bags = n_bags if n_bags != None else len(self.metadata)
        self.metadata = self.metadata.head(self.n_bags)
        # shuffling
        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True) # samples all rows in a random order
        self.train = train

        if n_train is None or n_test is None:
            n_train = int(np.floor(0.8 * self.n_bags))
            n_test = self.n_bags - n_train
        elif n_train + n_test > self.n_bags:
            raise ValueError(f'Not enough data for desired train/test split, max is {self.n_bags}')
        
        self.n_train = n_train
        self.n_test = n_test

        self.metadata_train = self.metadata.iloc[:self.n_train]
        self.metadata_test = self.metadata.iloc[self.n_train:self.n_train + self.n_test]

        # store the slides in temporary file (for faster access)
        try:
            with h5py.File(os.path.join(TMP_PATH, f"TCGA_PML_{TASK_ID}.h5"), "w-") as f: # if this fails, the file has already been written
                for i in tqdm(range(self.n_bags), desc="Creating temporary h5py file for faster data access."):
                    n_patches = get_bag_shape(self.slide_paths[self.metadata.iloc[i]["slide_id"]])[0]
                    ds = f.create_dataset(self.metadata.iloc[i]["slide_id"], (1, n_patches, 768), maxshape=(1, None, 768))
                    ds[0] = get_bag(self.slide_paths[self.metadata.iloc[i]["slide_id"]])
        except Exception as e:
            pass

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test
        
    def __str__(self):
        if self.train:
            return f" metadata_train: {self.metadata_train}\n n_train: {self.n_train}\n n_bags:{self.n_bags}\n"
        else :
            return f" metadata_test: {self.metadata_test}\n n_test: {self.n_test}\n n_bags:{self.n_bags}\n"

    def _get_bag(self, slide_id : str):
        with h5py.File(os.path.join(TMP_PATH, f"TCGA_PML_{TASK_ID}.h5"), "r") as f:
            bag = f[slide_id][0]
            
        return bag

    def _get_positive_bag_proportion(self):
        if self.train:
            sumt4_col = self.metadata_train["ajcc_pathologic_t"].apply(lambda x: 1 if x.startswith("T4") or x.startswith("T3") else 0)
            return sumt4_col.sum()/self.n_train
        else:
            sumt4_col = self.metadata_test["ajcc_pathologic_t"].apply(lambda x: 1 if x.startswith("T4") or x.startswith("T3") else 0)
            return sumt4_col.sum()/self.n_test

    def _get_label(self, i):
        if self.train:
            return np.array([1]) if self.metadata_train.iloc[i]["ajcc_pathologic_t"].startswith("T4") or self.metadata_train.iloc[i]["ajcc_pathologic_t"].startswith("T3") else np.array([0])
        else:
            return np.array([1]) if self.metadata_test.iloc[i]["ajcc_pathologic_t"].startswith("T4") or self.metadata_test.iloc[i]["ajcc_pathologic_t"].startswith("T3") else np.array([0])

    def __getitem__(self, i):
        # Return i-th bag

        if self.train:
            bag_id = self.metadata_train.iloc[i]["slide_id"]
        else:
            bag_id = self.metadata_test.iloc[i]["slide_id"]
        label = self._get_label(i)
        bag = self._get_bag(bag_id)
        
        if self.transformation:
            return self.transformation(bag, label)
        return bag, label