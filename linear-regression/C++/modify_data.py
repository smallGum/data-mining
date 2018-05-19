import pandas as pd
import numpy as np

def savePartition(data):
    print "solving dataset"
    df = pd.DataFrame(data)
    df.to_csv("newTest.csv", index=0, header=0, sep=' ')

data = pd.read_csv('test.csv')
del data["id"]

mat = data.as_matrix()

savePartition(mat)