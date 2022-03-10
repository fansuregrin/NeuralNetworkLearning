from dnn_utils import *


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