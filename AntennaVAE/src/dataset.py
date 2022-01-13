from torch.utils.data import Dataset
import torch

class dataset(Dataset):
    """\
        Description:
        ------------
            Create Pytorch Dataset

        Parameters:
        ------------
            counts: gene count. Type: numpy ndarrary
            anno: cell type annotations of the cells, of the shape the same as number of cells
        Return:
        ------------
            Dataset
        """
    def __init__(self, counts, anno = None):

        assert not len(counts) == 0, "Count is empty"
        self.counts = torch.FloatTensor(counts)
        if anno is None:
            self.anno = None
        else:
            self.anno = anno
                   
    def __len__(self):
        return self.counts.shape[0]
    
    def __getitem__(self, idx):
        if self.anno is not None:
            sample = {"count": self.counts[idx,:], "anno": self.anno[idx]}
        else:
            sample = {"count": self.counts[idx,:]}
        return sample