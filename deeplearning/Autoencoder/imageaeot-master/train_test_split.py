from sklearn.model_selection import train_test_split
import pandas as pd


trainch = pd.read_csv('/mountb/single_cell_flist/train_chall.csv')
valch = pd.read_csv('/mountb/single_cell_flist/val_chall.csv')
trainch.sample(frac = .03, random_state = 3).to_csv('/mountb/single_cell_flist/train_chall_med.csv')
valch.sample(frac = .03, random_state = 3).to_csv('/mountb/single_cell_flist/val_chall_med.csv')


