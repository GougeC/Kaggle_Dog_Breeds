import pandas as pd
import numpy as np 
import os

if __name__ == "__main__":
    labels = pd.read_csv('dogs/labels.csv')

    dic = {}
    for _id,breed in labels.values:
        if breed in dic:
            dic[breed].append(_id)
        else:
            dic[breed] = [_id]

    os.mkdir('train')
    for breed in dic:
        os.mkdir("train/{}".format(breed))

    for breed, id_list in dic.items():
        for _id in id_list:
            old = "dogs/train/{}.jpg".format(_id)
            new = "train/{}/{}.jpg".format(breed,_id)
            os.rename(old,new)


    os.mkdir('validation')
    for breed in dic:
        os.mkdir("validation/{}".format(breed))

    val_size = 0
    for breed, id_list in dic.items():
        n = len(id_list)//10
        val_size+=n
        print(breed, n)
        for _id in id_list[0:n]:
            old = "train/{}/{}.jpg".format(breed,_id)
            new = "validation/{}/{}.jpg".format(breed,_id)
            os.rename(old,new)