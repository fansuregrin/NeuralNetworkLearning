from dnn_utils import *
import os
from PIL import Image


def gen_imgs_from_dataset(dataset_path: str, img_save_folder: str, set_name: str, format: str = 'jpg'):        
    with h5py.File(dataset_path) as hf:
        imgs_arr = np.array(hf[set_name])
    i = 1
    for img_arr in imgs_arr[:]:
        img = Image.fromarray(img_arr, mode='RGB')
        img.save(f'{img_save_folder}/{i}.{format}')
        i += 1

def gen_datatest_from_images(img_folder: str, dataset_path: str):
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
                set_y_arr = np.array([[0]])
            else:
                img_arr = np.array(img).reshape((1, 64, 64, 3))
                set_x_arr = np.append(set_x_arr, img_arr, axis=0)
                set_y_arr = np.append(set_y_arr, [[0]], axis=0)
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
        my_image = np.array(image).reshape(1, 64*64*3).T
        my_predict = predict(my_image, [[1]], parameters)
    except Exception:
        my_predict = [[0]]
    return bool(np.squeeze(my_predict))


if __name__ == '__main__':
    filepath = input('your image:')
    if detect(filepath, 'test_para.h5'):
        print('it is a cat!')
    else:
        print('it is not a cat!')