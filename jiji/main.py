import torch
import pandas as pd


df = pd.read_csv("F:\PythonCode\data\\bank-additional-train.csv")
print(df.head())

te = torch.tensor(df.values)
print(te.shape)
