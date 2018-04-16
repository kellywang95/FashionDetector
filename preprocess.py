import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import torchvision
from matplotlib.pyplot import imshow

def convertImage(imageName):
    image = Image.open('Data/Img/'+ imageName).convert('L') 
    box = get_gt_bbox(imageName, df_bbox)
    image = image.crop(box)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image_arr = np.array(image)
    return image_arr
    
def get_gt_bbox(image_path_name, df_bbox):
    idx = np.where(df_bbox.image_name == image_path_name)[0][0]
    bbox = tuple(df_bbox.iloc[idx, 1:])
    return bbox

def preprocess(train_set):
    processed = []
    for image_name, category_label in zip(train_set['image_name'],train_set['category_label']):
        image = convertImage(image_name).reshape(784)
        row = np.append(image, category_label)
        processed.append(row)
    return np.array(processed)

if __name__ == "__main__":
    print("loading data...")
    data = pd.read_table('Data/Anno/list_category_img.txt',delim_whitespace=True, skiprows=[0])
    df_bbox = pd.read_table("Data/Anno/list_bbox.txt", delim_whitespace=True, skiprows=[0])

    print("sampling...")
    data = data.groupby('category_label').apply(pd.DataFrame.sample, frac=0.2).reset_index(drop=True)
    train = data.groupby('category_label').apply(pd.DataFrame.sample, frac=0.75).reset_index(drop=True)
    valid = data[~data.image_name.isin(train['image_name'])]

    print("preprocessing...")
    train_set = preprocess(train)
    valid_set = preprocess(valid)

    print("saving as numpy array...")
    np.save("train_set.npy", train_set)
    np.save("valid_set.npy", valid_set)