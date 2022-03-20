from dnn_utils import *
from PIL import Image
import os
import h5py


def load_parameters(path: str):
    """Load parameters from hdf5 file

    Generate a dictionary contained parameters from a specifical hdf5 file.

    Arguments:
    path -- The path to the hdf5 file where the parameters are stored.

    Returns:
    parameters -- A dictionary contained parameters (W1, b1, ..., WL, bL).
    """
    
    parameters = {}
    with h5py.File(path) as para_hd:
        for w in para_hd['weight']:
            parameters[w] = np.array(para_hd['weight'][w])
        for b in para_hd['bias']:
            parameters[b] = np.array(para_hd['bias'][b])
    
    return parameters

def load_train_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5')
    train_set_x_orig = np.array(train_dataset['train_set_x'])
    train_set_y_orig = np.array(train_dataset['train_set_y'])
    train_dataset.close()
    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255.0
    train_set_y = train_set_y_orig.reshape(train_set_y_orig.shape[0], 1).T

    return train_set_x, train_set_y

def load_dataset(h5_file_path: str):
    dataset = h5py.File(h5_file_path)
    set_x_orig = np.array(dataset['set_x'])
    set_y_orig = np.array(dataset['set_y'])
    dataset.close()
    set_x = set_x_orig.reshape(set_x_orig.shape[0], -1).T/255.0 
    set_y = set_y_orig.reshape(set_y_orig.shape[0], 1).T

    return set_x, set_y

def load_test_dataset():
    test_dataset = h5py.File('datasets/test_catvnoncat.h5')
    test_set_x_orig = np.array(test_dataset['test_set_x'])
    test_set_y_orig = np.array(test_dataset['test_set_y'])
    test_dataset.close()
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T/255.0
    test_set_y = test_set_y_orig.reshape(test_set_y_orig.shape[0], -1).T

    return test_set_x, test_set_y

def gen_imgs_from_dataset(dataset_path: str, img_save_folder: str, set_name: str, format: str = 'jpg'):        
    with h5py.File(dataset_path) as hf:
        imgs_arr = np.array(hf[set_name])
    i = 1
    for img_arr in imgs_arr[:]:
        img = Image.fromarray(img_arr, mode='RGB')
        img.save(f'{img_save_folder}/{i}.{format}')
        i += 1

def gen_datatest_from_same_class_images(img_folder: str, dataset_path: str, class_id: int):
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    hf = h5py.File(dataset_path, 'a')
    set_x_arr = None
    set_y_arr = None
    for img_name in os.listdir(img_folder):
        try:
            img = Image.open(f"{img_folder}/{img_name}")
            if img.mode != 'RGB':
                img.convert('RGB')
            if img.size != (64, 64):
                img.resize((64, 64))
            if set_x_arr is None:
                set_x_arr = np.array(img).reshape((1, 64, 64, 3))
                set_y_arr = np.array([[class_id]])
            else:
                img_arr = np.array(img).reshape((1, 64, 64, 3))
                set_x_arr = np.append(set_x_arr, img_arr, axis=0)
                set_y_arr = np.append(set_y_arr, [[class_id]], axis=0)
            print(f"put {img_name} into dataset...")
        except Exception:
            pass
    hf.create_dataset('set_x', set_x_arr.shape, data=set_x_arr)
    hf.create_dataset('set_y', set_y_arr.shape, data=set_y_arr)
    hf.close()

def detect(filepath, para_path):
    parameters = {}
    with h5py.File(para_path) as hf:
        for w in hf['weight']:
            parameters[w] = np.array(hf['weight'][w])
        for b in hf['bias']:
            parameters[b] = np.array(hf['bias'][b])
    with Image.open(filepath) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = img.resize((64, 64))
    try:
        my_image = np.array(image).reshape(1, 64*64*3).T/255.0
        my_predict = identify(my_image, parameters)
    except Exception:
        my_predict = [[0]]
    return bool(np.squeeze(my_predict))


if __name__ == '__main__':
    pass
    # gen_datatest_from_same_class_images('/run/media/fansuregrin/SharedData/Linnaeus 5 64X64/train/berry', 'berry_2.h5', 0)
    # gen_datatest_from_same_class_images('/run/media/fansuregrin/SharedData/Linnaeus 5 64X64/train/bird', 'birds_2.h5', 0)
    # gen_datatest_from_same_class_images('/run/media/fansuregrin/SharedData/Linnaeus 5 64X64/train/flower', 'flower_2.h5', 0)
    # gen_datatest_from_same_class_images('/run/media/fansuregrin/SharedData/Linnaeus 5 64X64/train/dog', 'dogs_2.h5', 0)