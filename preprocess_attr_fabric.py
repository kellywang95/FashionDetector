import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import torchvision
from matplotlib.pyplot import imshow
import argparse

def convertImage(imageName, args):
    image = Image.open('Data/Img/'+ imageName)
    if not args['rgb']:
        image = image.convert('L') 
    box = get_gt_bbox(imageName, df_bbox)
    image = image.crop(box)
    image = image.resize((args['resize'], args['resize']), Image.ANTIALIAS)
    image_arr = np.array(image)
    return image_arr
    
def get_gt_bbox(image_path_name, df_bbox):
    idx = np.where(df_bbox.image_name == image_path_name)[0][0]
    bbox = tuple(df_bbox.iloc[idx, 1:])
    return bbox

def get_fabrics(att_data):
    att = pd.read_table('Data/Anno/list_attr_cloth.txt', skiprows=[0, 1], names = ['name', 'type'])
    att['column_new'] = att['name'].str.split(' ')

    new_att = att_data.iloc[:, [0]]
    fabrics = []

    for i in range(1000):
        if att['column_new'].iloc[i][-1] is '2':
            fabrics.append(att['name'].iloc[i])
            new_att = new_att.join(att_data.ix[ :,i])

    fabrics = np.array(fabrics)
    np.save("fabrics.npy", fabrics)
    return new_att


def preprocess(train_set, args):
    processed = []
    labels = np.zeros([train_set.shape[0], train_set.shape[1]])
    ones = np.where(train_set == 1)
    labels[ones] = 1
    labels = labels[:, 1:].astype(int)
    for i in range(len(train_set)):
        image_name = train_set.iloc[i][0]
        image = convertImage(image_name, args)
        processed.append(image)
    return np.array(processed), labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--rgb', default=False, type=bool)
    parser.add_argument('--resize', default=28, type=int)
    parser.add_argument('--sample_fraction', default=0.2, type=float)
    parser.add_argument('--train_fraction', default=0.75, type=float)
    args = parser.parse_args()

    print("loading data...")
    data = pd.read_table('Data/Anno/list_attr_img.txt',delim_whitespace=True, skiprows=[0, 1], header=None)
    data = get_fabrics(data)
    df_bbox = pd.read_table("Data/Anno/list_bbox.txt", delim_whitespace=True, skiprows=[0])

    print("sampling...")
    data = data.sample(frac=args['sample_fraction']).reset_index(drop=True)
    train = data.sample(frac=args['train_fraction']).reset_index(drop=True)
    valid = data[~data[0].isin(train[0])]
    
    print(train.shape, valid.shape)
    
          
    print("preprocessing...")
    train = get_fabrics(train)
    train = get_fabrics(valid)
    
    train_set, train_labels = preprocess(train, args)
    valid_set, valid_labels = preprocess(valid, args)
    
    print("saving as numpy array...")
    np.save("train_set.npy", train_set)
    np.save("valid_set.npy", valid_set)
    np.save("train_labels.npy", train_labels)
    np.save("valid_labels.npy", valid_labels)
