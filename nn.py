from dnn_utils import *
import matplotlib.pyplot as plt


def train_and_test_dnn(para_path: str, train_X, train_Y, test_X, test_Y, layers_dims, learning_rate, 
                       graph_name = 'costs_curve', num_iterations=3000, lambd=0, keep_prob=1.0, print_cost=True):
    
    # training parameters
    print("\033[92m" + f"training para: learning_rate={learning_rate}, iterations={num_iterations}, lambda={lambd}, keep_prob={keep_prob}, \
layers_dims: {' '.join([str(nes) for nes in layers_dims])}" + "\033[0m")
    parameters, costs = L_layer_model(train_X, train_Y, layers_dims, learning_rate=learning_rate, num_iterations=num_iterations, 
                               lambd=lambd, keep_prob=keep_prob, print_cost=print_cost)
    
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
    plt.figure(figsize=(12,9))
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x10)')
    layers_neurons = ' -> '.join([str(nes) for nes in layers_dims])
    plt.title("Learning rate = " + str(learning_rate) + "\nlambda = " + str(lambd) + \
        "\nneurons per layer: " + layers_neurons)
    plt.text(len(costs), costs[-1], f'last cost: {costs[-1]}')
    plt.text(len(costs)//2, costs[0], 'train set accuracy: {:.3f}\ntest set accuracy: {:.3f}'.format(train_accu, test_accu))
    plt.savefig(f'{graph_name}.png', format='png')


def test_dnn(para_path: str, X, Y):
    parameters = load_parameters(para_path)
    pred, accu = predict(X, Y, parameters)
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
    # train_X, train_Y = load_train_dataset()
    # test_X, test_Y = load_test_dataset()
    # 
    np.random.seed(1)
    train_X = np.random.rand(10, 5)
    train_Y = np.array([[1, 0, 0, 1, 0]])
    test_X = np.random.rand(10, 2)
    test_Y = np.array([[1, 0]])

    # set neurons per layer
    # layers_dims = (train_X.shape[0], 20, 10, 5, 1)
    layers_dims = (train_X.shape[0], 4, 1)
    # layers_dims = [train_X.shape[0], 10, 5, 1]
    
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
    # train_and_test_dnn('para_dropout_01.h5', train_X, train_Y, test_X, test_Y, layers_dims, learning_rate=0.007, 
                        # graph_name='costs_graph_dropout_01', keep_prob=0.9)
    train_and_test_dnn('check_para.h5', train_X, train_Y, test_X, test_Y, layers_dims, learning_rate=0.007,
                        graph_name='costs_check', num_iterations=10, lambd=0.8, keep_prob=0.8, print_cost=False)