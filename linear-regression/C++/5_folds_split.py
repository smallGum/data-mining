import pandas as pd
import numpy as np

def savePartition(data, num):
    print "solving " + str(num) + " dataset"
    df = pd.DataFrame(data)
    df.to_csv("part" + str(num) + ".csv", index=0, header=0, sep=' ')

data = pd.read_csv('train.csv')
del data["id"]

mat = data.as_matrix()

savePartition(mat[0:5000][:], 1)
savePartition(mat[5000:10000][:], 2)
savePartition(mat[10000:15000][:], 3)
savePartition(mat[15000:20000][:], 4)
savePartition(mat[20000:25000][:], 5)