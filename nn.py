from dnn_utils import *
from help_utils import *
import matplotlib.pyplot as plt


def train_and_test_dnn(para_path: str, train_X, train_Y, test_X, test_Y, layers_dims, learning_rate, 
                       graph_name = 'costs_curve', num_iterations = 3000, lambd = 0, keep_prob = 1.0, 
                       print_cost = True, check_back_prop = False, step = 100, user_mini_batch = False,
                       mini_batch_size = 64):
    
    # training parameters
    print("\033[92m" + f"training para: learning_rate={learning_rate}, iterations={num_iterations}, lambda={lambd}, \
keep_prob={keep_prob}, layers_dims: {' '.join([str(nes) for nes in layers_dims])} samples number: {train_X.shape[1]}" + "\033[0m")

    parameters, costs = L_layer_model(train_X, train_Y, layers_dims, learning_rate = learning_rate, 
                                        num_iterations = num_iterations, lambd = lambd, keep_prob = keep_prob, 
                                        print_cost = print_cost, check_back_prop = check_back_prop,
                                        step = step, use_mini_batch = user_mini_batch, mini_batch_size=mini_batch_size)

    # save parameters
    print(f"saving parameters into {para_path}...")
    with h5py.File(para_path, 'w') as hf:
        hf.create_group('weight')
        hf.create_group('bias')
        for i in range(len(parameters)//2):
            hf['weight'].create_dataset(f"W{i+1}", data=parameters[f"W{i+1}"])
            hf['bias'].create_dataset(f"b{i+1}", data=parameters[f"b{i+1}"])
    
    # predict and get accuracy
    _, train_accu = predict(train_X, train_Y, parameters)
    _, test_accu = predict(test_X, test_Y, parameters)
    print(f"train set accuracy: {str(train_accu)}")
    print(f"test set accuracy: {str(test_accu)}")

    # save info graph
    print(f"saving learing graph into {graph_name}...")
    plt.figure(figsize=(15,10))
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x{:d})'.format(step))
    layers_neurons = ' -> '.join([str(nes) for nes in layers_dims])
    plt.title("Learning rate = " + str(learning_rate) + "\nlambda = " + str(lambd) + ", keep_prob = " + str(keep_prob) + \
              "\nneurons per layer: " + layers_neurons)
    left, right, bottom, top = plt.axis()
    plt.text(len(costs), costs[-1], 'last cost: {:.6f}'.format(costs[-1]))
    plt.text(right, top, 'train_set: {}\ntest_set: {}\nmini-batch size: {}\n'.format(train_X.shape, train_Y.shape, mini_batch_size) + 
                'train set accuracy: {:.3f}\ntest set accuracy: {:.3f}'.format(train_accu, test_accu), 
                horizontalalignment='right', verticalalignment='top')
    plt.savefig(f'{graph_name}.png', format='png')


def test_dnn(para_path: str, X, Y):
    parameters = load_parameters(para_path)
    pred, accu = predict(X, Y, parameters)
    print(f'accuracy: {str(accu)}')
    mis = (pred == Y)
    mis_index = np.squeeze(np.argwhere(mis == False)[:, 1])
    imgs = X[:, mis_index].T.reshape((-1, 64, 64, 3))
    for i in range(imgs.shape[0]):
        plt.subplot(1, imgs.shape[0], i+1)
        plt.title(f"image {mis_index[i]+1}", fontsize=5)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.5)
    plt.savefig('mis.jpg', dpi=1000)


if __name__ == '__main__':
    
    # load set
    train_X, train_Y = load_dataset('datasets/dataset_2.h5')
    x, y = load_train_dataset()
    train_X = np.concatenate((train_X, x), axis=1)
    train_Y = np.concatenate((train_Y, y), axis=1)
    test_X, test_Y = load_dataset('datasets/dataset_1.h5')
    t_x, t_y = load_test_dataset()
    test_X = np.concatenate((test_X[:, 0:150], t_x), axis=1)
    test_Y = np.concatenate((test_Y[:, 0:150], t_y), axis=1)
    print('train_X: ' + str(train_X.shape))
    print('train_Y: ' + str(train_Y.shape))
    print('test_X: ' + str(test_X.shape))
    print('test_Y: ' + str(test_Y.shape))
    # 
    # np.random.seed(1)
    # train_X = np.random.rand(10, 5)
    # train_Y = np.array([[1, 0, 0, 1, 0]])
    # test_X = np.random.rand(10, 2)
    # test_Y = np.array([[1, 0]])

    # set neurons per layer
    # layers_dims = (train_X.shape[0], 20, 10, 5, 1)
    # layers_dims = (train_X.shape[0], 10, 1)
    layers_dims = [train_X.shape[0], 10, 5, 1]
    
    # gen learning rates list for tuning
    # rates = set()
    # while len(rates) != 10:
        # rates.add(np.around(10**(-3*np.random.rand()), 3))
        # rand_rate = np.around(10**-2 * np.random.rand(), 3)
        # if rand_rate != 0.0:
            # rates.add(rand_rate)
    # rates = sorted(rates)
    
    # tuning learing rates
    # i = 0
    # for rate in rates:
        # train_and_test_dnn('learning_para_{:03d}.h5'.format(i+1), train_X, train_Y, test_X, test_Y, 
                            # layers_dims, learning_rate=rate, graph_name='learning_graph_{:03d}'.format(i+1))
        # i += 1;
        
    # dir = 'layers_neurons_tuning_4'
    # for i in range(6):
        # layers_dims[2] = (i+1)*2
        # train_and_test_dnn('{}/para_{:02d}.h5'.format(dir, i+1), train_X, train_Y, test_X, test_Y, layers_dims, 
                        #    learning_rate=0.007, graph_name='{}/costs_graph_{:02d}'.format(dir, i+1))
    
    # for i in range(4):
        # train_and_test_dnn('para_dropout_0{}.h5'.format(i+6), train_X, train_Y, test_X, test_Y, layers_dims, learning_rate=0.007, 
                        # graph_name='costs_graph_dropout_0{}'.format(i+6), keep_prob=0.1*(4-i))
    
    # train_and_test_dnn('check_para.h5', train_X, train_Y, test_X, test_Y, layers_dims, learning_rate=0.007,
                        # graph_name='costs_check', num_iterations=10, lambd=0.8, keep_prob=0.8, print_cost=False)
    
    # test_dnn('layers_neurons_tuning_3/para_05.h5', test_X, test_Y)

    train_and_test_dnn('mini_batch/mini_batch_para03.h5', train_X, train_Y, test_X, test_Y, layers_dims,
                                  learning_rate=0.004, graph_name='mini_batch/costs_graph03', user_mini_batch=True,
                                  mini_batch_size=512, lambd=0, num_iterations=1000)