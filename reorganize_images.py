import pandas as pd
import numpy as np 
import os
import sys

if __name__ == "__main__":
    labels = pd.read_csv('dogs/labels.csv')
    if len(sys.argv) > 1:
        make_val = sys.argv[1]
        if make_val == "True":
            make_val = True
        else:
            make_val = False
    else:
        make_val = False
        
    dic = {}
    for _id,breed in labels.values:
        if breed in dic:
            dic[breed].append(_id)
        else:
            dic[breed] = [_id]
    prefix = ''
    if not make_val:
        prefix = 'submission_'
    os.mkdir(prefix+'train')

    for breed in dic:
        os.mkdir("{}train/{}".format(prefix,breed))

    for breed, id_list in dic.items():
        for _id in id_list:
            old = "./dogs/train/{}.jpg".format(_id)
            new = "{}train/{}/{}.jpg".format(prefix,breed,_id)
            os.rename(old,new)

    if make_val:
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