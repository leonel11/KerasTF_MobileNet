import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # no display mode
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
from keras.applications.mobilenet import MobileNet
from keras.optimizers import TFOptimizer
from keras import backend as K # backend of Keras
from keras.utils import multi_gpu_model, to_categorical
import scipy.misc


np.random.seed(1)
K.set_image_dim_ordering('tf') # use Tensorflow backend


# parameters of learning for the network
EPOCHES = 2 # how many times the whole train set will be shown for model
BATCH_SIZE = 128 # count of train samples, which are shown to optimizer before updating weights of network
IMG_ROWS, IMG_COLS = 224, 224 # size of images (at least 32x32)
IMG_CHANNELS = 1 # colour channels of images
CLASSES = 1000

# parameters of optimizer
LEARNING_RATE = 0.01
DECAY = 0.9
MOMENTUM = 0.9


def get_batch(data_path, data_list, batch_size):
    '''
    Form batch for training or testing
    @param data_path: path to the data folder
    @param data_list: pandas list of images and labels
    @param batch_size: size of formed batch
    @return: batch of images, coresponding classes
    '''
    indeces = np.random.randint(data_list.shape[0], size=batch_size) # indices of pictures of formed batch
    for i, idx in enumerate(indeces):
        label = np.array([int(data_list.iloc[idx][1])])
        img = scipy.misc.imresize(scipy.misc.imread(os.path.join(data_path, data_list.iloc[idx][0]),
                                                    flatten=True, mode='LA'), (224, 224)) # read reshaped image
        # image -> numpy normalized array
        arr_img = np.array(img).astype('float32') / 255.0
        arr_img = arr_img[np.newaxis, :, :, np.newaxis]
        # add data to batch
        if i == 0:
            x = arr_img
            y = label
        else:
            x = np.concatenate([x, arr_img])
            y = np.concatenate([y, label])
    return x, to_categorical(y, num_classes=CLASSES)


def get_gpus(gpus):
    '''
    Get numbers of GPU devices
    @param gpus: numbers of GPU-videocards for training the network
    @return: list with drivers of GPU-devices
    '''
    return list(map(int, gpus.split(',')))


def get_optimizer():
    '''
    Get optimizer for learning the network
    @return: optimizing function
    '''
    optimizer = TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,
                                                      decay=DECAY,
                                                      momentum=0.9)) # Tensorflow RMSProp
    return optimizer


def get_model(optimizer, gpus):
    '''
    Get compiled MobileNet models according to optimizer
    @param optimizer: optimizing function for training the network
    @return: compiled MobileNet models: for parallel launch and the network
    '''
    model = MobileNet(input_shape=(IMG_ROWS, IMG_ROWS, IMG_CHANNELS),
                      alpha=1.0, # control the width of the network
                      include_top=True, # including FC-layers at the network
                      weights=None,
                      classes=CLASSES)
    '''
    Architecture of MobileNet v1:
    [Conv 3x3] -> [BatchNorm] -> [ReLU] ->
    -> (x13) [Deepthwise Conv 3x3] -> [BatchNorm] -> [ReLU] -> [Conv 1x1] -> [BatchNorm] -> [ReLU]
    '''
    model.summary() # print the architecture of model
    # TODO: optimize work on many GPUs
    if len(gpus) == 1:
        with K.tf.device('/gpu:{}'.format(gpus[0])):
            parallel_model = model
    else:
        with K.tf.device('/cpu:0'):
            kernel_model = model
        parallel_model = multi_gpu_model(kernel_model, gpus=gpus)
    parallel_model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy', # for multiclass classification
                           metrics=['accuracy'])
    return parallel_model, model


def train(model, data_path, train_list_file, val_list_file):
    '''
    Train the network.
    @param model: network
    @param X_train: train+validation data
    @param Y_train: train+validation labels
    @return: history of training
    '''
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    # check the rightness of data
    if not os.path.isdir(data_path) or not os.path.isdir(train_path) or not os.path.join(val_path):
        print('The dataset doesn\'t have the right arragement (\'train\' and \'val\' data with proper data)')
        return None
    train_list = pd.read_csv(train_list_file, sep=' ')
    val_list = pd.read_csv(val_list_file, sep=' ')
    count_batches = int(train_list.shape[0]/BATCH_SIZE)
    print('\nStart training...\n')
    # manual implementation of training and validation
    for ep in range(EPOCHES):
        print('\nEpoch {}/{}\n'.format(ep+1, EPOCHES))
        train_loss = []
        train_acc = []
        # Training a epoch
        for b in range(count_batches):
            x_train, y_train = get_batch(train_path, train_list, BATCH_SIZE)
            hist = model.train_on_batch(x_train, y_train)
            train_loss.append(hist[0])
            train_acc.append(hist[1])
            print('\tBatch {}/{}: \t\t loss: {},\t acc: {}'.format(b+1, count_batches, train_loss[-1], train_acc[-1]))
        history['loss'].append(np.asarray(train_loss).mean())
        history['acc'].append(np.asarray(train_acc).mean())
        # Validation a epoch
        x_val, y_val = get_batch(val_path, val_list, BATCH_SIZE)
        hist = model.test_on_batch(x_val, y_val)
        history['val_loss'].append(hist[0])
        history['val_acc'].append(hist[1])
        print('\nTest loss: {},\t test accuracy: {}'.format(history['val_loss'][-1], history['val_acc'][-1]))
    print('Training finished!')
    return history


def save_model(results_path, model):
    '''
    Save the model and its weights to files
    @param model: model for saving
    '''
    open(os.path.join(results_path, 'MobileNetV1.json'), 'w').write(model.to_json()) # save the architecture to json
    model.save_weights(os.path.join(results_path, 'MobileNetV1.h5'), overwrite=True) # save weights of network
    print('\nModel saved!\n')


def create_folder_results():
    '''
    Create the folder for saving results of training the network
    @return: path to folder
    '''
    folder_name = datetime.now().strftime('___%d.%m.%Y___%H.%M.%S')
    results_path = os.path.join('../results', folder_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path

 
def plot_results(results_path, model, history):
    '''
    Plot results of training
    @param history: history of training
    '''
    if not history:
        return
    # history of accuracy
    # build plot
    plt.figure(figsize=(10, 4))
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.plot(range(1, EPOCHES + 1), history['acc'])
    plt.plot(range(1, EPOCHES + 1), history['val_acc'])
    plt.legend(['train', 'test'])
    plt.title('History of accuracy')
    plt.savefig(os.path.join(results_path, 'accuracy.png')) # save plot
    plt.clf() # clear the previous shown history
    # history of loss
    # build plot
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.grid(True)
    plt.plot(range(1, EPOCHES + 1), history['loss'])
    plt.plot(range(1, EPOCHES + 1), history['val_loss'])
    plt.legend(['train', 'test'])
    plt.title('History of loss')
    plt.savefig(os.path.join(results_path, 'loss.png'))  # save plot


def launch_network(gpus, data_path, train_list_file, val_list_file):
    '''
    Launch the training of network
    @param gpus: GPU devices, on which the training will be implemented
    @param data_path: path to selection
    @param train_list_file: file contained train data and its labels
    @param val_list_file: file contained validation data and its labels
    '''
    optimizer = get_optimizer()
    parallel_model, model = get_model(optimizer, gpus)
    history = train(parallel_model, data_path, train_list_file, val_list_file)
    results_path = create_folder_results()
    save_model(results_path, model)
    plot_results(results_path, model, history)


def init_argparse():
    '''
    Initialize argparser
    @return: parsed command-line arguments of cript
    '''
    parser = ArgumentParser(description='MobileNet v.1 network')
    parser.add_argument(
        '-gpu',
        '--gpus',
        nargs='?',
        help='GPU device numbers (0,1,3,...)',
        default='0',
        type=str)
    parser.add_argument(
        '-data',
        '--data_path',
        nargs='?',
        help='Path to samples',
        default='/mobilenet/CLS-LOC_multiscale',
        type=str)
    parser.add_argument(
        '-train_list',
        '--train_list',
        nargs='?',
        help='Path to file of train list (structure: 2 columns, files labels)',
        default='/mobilenet/train_list_multiscale',
        type=str)
    parser.add_argument(
        '-val_list',
        '--validation_list',
        nargs='?',
        help='Path to file of validation list (structure: 2 columns, files labels)',
        default='/mobilenet/test_list_multiscale',
        type=str)
    return parser


def main():
    '''
    Main function
    '''
    args = init_argparse().parse_args()
    gpus = get_gpus(args.gpus)
    data_path = args.data_path
    train_list_file = args.train_list
    val_list_file = args.validation_list
    launch_network(gpus, data_path, train_list_file, val_list_file)


if __name__ == '__main__':
    main()