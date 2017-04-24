import numpy as np
import read_soft
import learning
import random
import pickle


def random_shuffle(data, label):
    label = np.transpose([label])
    meta_data = np.c_[data, label]
    shape_meta = np.shape(meta_data)
    np.random.seed(0)
    np.random.shuffle(meta_data)
    data = meta_data[:,:shape_meta[1]-1]
    y = meta_data[:,shape_meta[1]-1]
    print(80*'-')
    print('Show the labels after permutations')
    print(y)
    return (data, y)

def main():
    DATA_EXIST = 0
    GDS_ONLY = 0
    NUM_FEATURE = 500

    # GDS_ONLY = 1
    # NUM_FEATURE = 100
    if (DATA_EXIST == 0):
        ids, data, label = read_soft.PreprocessData(GDS_ONLY, NUM_FEATURE)
        data, label = random_shuffle(data,label)
        with open('data/data.pkl', 'wb') as f:
            pickle.dump((data,label),f)
    else:
        with open('data/data.pkl','rb') as f:
            (data,label) = pickle.load(f)


    N_TRAINING = int(len(label) / 4 * 3)
    N_TEST = len(label) - N_TRAINING
    print('Take 3/4 samples to train:{} \n and 1/4 samples to test {}'.format(
        N_TRAINING,
        N_TEST
    ))
    x_train = data[0:N_TRAINING,:]
    y_train = label[0:N_TRAINING]
    x_test = data[N_TRAINING:,:]
    y_test = label[N_TRAINING:]
    # print(np.shape(x_train))
    # print(np.shape(x_test))
    # learning.svm_learning(data, label)
    # learning.svm_learning(data, label, x_train,y_train,x_test,y_test)
    # learning.random_forest_learning(data, label, x_train, y_train, x_test, y_test)
    learning.nn_learning(data, label, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
