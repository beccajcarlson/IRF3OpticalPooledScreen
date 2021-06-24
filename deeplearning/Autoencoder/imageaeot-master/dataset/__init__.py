from .dataset import CellImageDataset, CellImageDataset46, CellImageDatasetwithTargets

dataset_dict = {'default': CellImageDataset,  '46': CellImageDataset46, 'labeled': CellImageDatasetwithTargets}
