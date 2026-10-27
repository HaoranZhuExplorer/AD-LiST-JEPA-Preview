import os
import yaml
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class LyftDataset(Dataset):
    """Occ4D-Lyft dataset class."""
    def __init__(self, root_dir, split='train', p_pre=10, p_post=10, semantic=False):
        """
        Args:
            root_dir (string): Directory with all the data.
            split (string): 'train' or 'test' split.
            p_pre (int): Number of frames before the current frame.
            p_post (int): Number of frames after the current frame.
        """
        self.root_dir = root_dir
        if split == 'train':
            self.split = list(range(120))
        elif split == 'valid':
            self.split = list(range(120, 150))
        elif split == 'test':
            self.split = list(range(150, 180))
        else:
            raise ValueError('Invalid split name: {}'.format(split))
        
        self.p_pre = p_pre + 1 # +1 for the current frame
        self.p_post = p_post + 1
        self.semantic = semantic
        
        with open("configs/lyft.yaml", 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self._load_data()

    def _load_data(self):
        self.data_path = []

        voxel_root = os.path.join(self.root_dir, "voxel")
    
        for seq_id in self.split:
            seq_str = f"{seq_id:04d}"
            seq_dir = os.path.join(voxel_root, seq_str)
            if not os.path.isdir(seq_dir):
                continue

            # find all *_input.npy in this sequence
            for input_path in sorted(glob(os.path.join(seq_dir, "*_input.npy"))):
                name = os.path.basename(input_path)
                if not name.endswith("_input.npy"):
                    continue
                sample_id = name[:-10]  # strip "_input.npy", e.g. "0069"

                label_path   = os.path.join(seq_dir, f"{sample_id}_label.npy")
                invalid_path = os.path.join(seq_dir, f"{sample_id}_invalid.npy")

                # only keep if all three exist
                if os.path.isfile(label_path) and os.path.isfile(invalid_path):
                    # absolute stub: "/Occ4D-Lyft/voxel/0000/0069"
                    stub = os.path.join(os.path.abspath(seq_dir), sample_id)
                    self.data_path.append(stub)


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data_label = np.load(self.data_path[idx]+"_label.npy")
        label = data_label[:self.p_post]
        data_input_arr = np.load(self.data_path[idx]+"_input.npy")
        data_input_bits = np.unpackbits(data_input_arr)
        data_input = data_input_bits.reshape(11, 512, 256, 32) # to do: shape hard coding for now
        input = data_input[-self.p_pre:, :, :, :]

        data_invalid_arr = np.load(self.data_path[idx]+"_invalid.npy")
        data_invalid_bits = np.unpackbits(data_invalid_arr)
        data_invalid = data_invalid_bits.reshape(11, 512, 256, 32) # to do: shape hard coding for now
        invalid = data_invalid[:self.p_post, :, :, :]

        input_tensor = torch.tensor(input, dtype=torch.float32)
        label_tensor = torch.tensor(label)
        invalid_tensor = torch.from_numpy(invalid)
        #print("debug", torch.sum(input_tensor), torch.sum(label_tensor), torch.sum(invalid_tensor))
        return input_tensor, label_tensor, invalid_tensor

    def get_voxel_size(self):
        X, Y, Z = 512, 256, 32 # to do: shape hard coding for now
        T = self.p_post
        return X, Y, Z, T

    def get_class_fc(self):
        if self.semantic:
            return self.config['label_fc']
        else:
            fc_dict = self.config['label_fc']
            for i in range(2, len(fc_dict)):
                fc_dict[1] += fc_dict[i]
                del fc_dict[i]
            return fc_dict