__author__ = 'Maria Khodorchenko'
import pandas as pd
import numpy as np
import csv

#matrix = csv.reader('prepared.csv')

matrix = pd.read_csv('prepared.csv', index_col =False)
print(matrix.head)

