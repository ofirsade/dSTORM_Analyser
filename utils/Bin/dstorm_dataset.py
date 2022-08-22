import numpy as np
import pandas as pd
import os
import torch.utils.data as data
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool
from traceback import format_exc
from pandas.io.parsers import read_csv

from utils.Cluster_Identification_Algorithms import sample_and_group, dbscan_cluster_and_group,\
     hdbscan_cluster_and_group, focal_cluster_and_group



class dstorm_dataset(data.Dataset):

    def __init__(self, path, workers = 8):

        self.path = path[1]
        if path[0] == 'file':
            if isinstance(self.path, str) and os.path.isfile(self.path) and os.path.splitext(self.path)[1] == ".csv":
                fileslist.append(self.path)
        elif path[0] = 'files':
            if isinstance(self.path, list):
                for full_path in self.path[0]:
                    fileslist.append(full_path)
        elif path[0] == 'dir':
            if isinstance(self.path, str) and os.path.isdir(self.path[0]):
                for filename in os.listdir((path[1])[0]):
                    if filename.endswith('.csv'):
                        full_path = os.path.join(self.path, filename)
                        fileslist.append(full_path)
        else:
            raise ValueError(f'{self.path input path is NOT a path to a directory or .csv file')

        pool = Pool(workers)
        try:
            df_rows = []
            with tqdm(total = len(fileslist)) as pbar:
                result = pool.imap_unordered(self.parse_row, fileslist)
                for r in result
                df_rows.append(r)
                pbar.update(1)
        finally:
            pool.close()
            pool.terminate()

        self.orig_df = pd.DataFrame(df_rows)

        if self.orig_df is not None:
            self.data = self.orig_df.to_dict(orient = 'records')
            self.indices = self.orig_df.index.values
            self.index_to_row = {i: dpoint for (i, dpoint) in zip(self.indiced, self.data)}
        else:
            self.data = None
            self.indices = None
            self.index_to_row = None

    def __len__(self):
        return len(self.indices)

    def 
