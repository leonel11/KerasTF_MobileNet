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
from keras.callbacks import ModelCheckpoint
from multiprocessing import cpu_count
import scipy.misc


np.random.seed(1)
K.set_image_dim_ordering('tf') # use Tensorflow backend


# parameters of learning for the network
EPOCHES = 10 # how many times the whole train set will be shown for model
BATCH_SIZE = 128 # count of train samples, which are shown to optimizer before updating weights of network
IMG_ROWS, IMG_COLS = 224, 224 # size of images (at least 32x32)
IMG_CHANNELS = 1 # colour channels of images
CLASSES = 1000

# folders for savings
RESULTS_DIR = '../results'
WEIGHTS_DIR = 'weights'


# parameters of optimizer
LEARNING_RATE = 0.01
DECAY = 0.9
MOMENTUM = 0.9


def generator(data_path, data_list, batch_size):
    """
    Form batch for training or testing
    @param data_path: path to the data folder
    @param data_list: pandas list of images and labels
    @param batch_size: size of formed batch
    @return:
    """
    indeces = np.random.randint(data_list.shape[0], size=batch_size) # indices of pictures of formed batch
    x = np.zeros((batch_size, IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    y = np.zeros((batch_size, 1))
    for i, idx in enumerate(indeces):
        label = int(data_list.iloc[idx][1])
        img = scipy.misc.imresize(scipy.misc.imread(os.path.join(data_path, data_list.iloc[idx][0]),
                                                    flatten=True, mode='LA'), (IMG_ROWS, IMG_COLS)) # read reshaped image
        # image -> numpy normalized array
        arr_img = np.array(img).astype('float32') / 255.0
        arr_img = arr_img[np.newaxis, :, :, np.newaxis]
        x[i] = arr_img
        y[i] = label
    yield x, to_categorical(y, num_classes=CLASSES)


def get_optimizer():
    """
    Get optimizer for learning the network
    @return: optimizing function
    """
    optimizer = TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,
                                                      decay=DECAY,
                                                      momentum=0.9)) # Tensorflow RMSProp
    return optimizer


def get_model(optimizer, gpus):
    """
    Get compiled MobileNet models according to optimizer
    @param optimizer: optimizing function for training the network
    @return: compiled MobileNet model
    """
    '''
    Architecture of MobileNet v1:
    [Conv 3x3] -> [BatchNorm] -> [ReLU] ->
    -> (x13) [Deepthwise Conv 3x3] -> [BatchNorm] -> [ReLU] -> [Conv 1x1] -> [BatchNorm] -> [ReLU]
    '''
    # TODO: optimize work on many GPUs
    with K.tf.device('/cpu:0'):
        kernel_model = MobileNet(input_shape=(IMG_ROWS, IMG_ROWS, IMG_CHANNELS),
                                 alpha=1.0, # control the width of the network
                                 include_top=True, # including FC-layers at the network
                                 weights=None,
                                 classes=CLASSES)
    parallel_model = multi_gpu_model(kernel_model, gpus=gpus)
    parallel_model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy', # for multiclass classification
                           metrics=['accuracy'])
    return parallel_model


def train(model, data_path, train_list_file, val_list_file):
    """
    Train the network.
    @param model: network
    @param X_train: train+validation data
    @param Y_train: train+validation labels
    @return: history of training
    """
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    # check the rightness of data
    if not os.path.isdir(data_path) or not os.path.isdir(train_path) or not os.path.join(val_path):
        print('The dataset doesn\'t have the right arragement (\'train\' and \'val\' data with proper data)')
        return None
    train_list = pd.read_csv(train_list_file, sep=' ')
    val_list = pd.read_csv(val_list_file, sep=' ')
    count_batches = int(train_list.shape[0]/BATCH_SIZE)
    print('\nStart training...\n')
    history = model.fit_generator(
        generator(data_path, train_list, BATCH_SIZE),
        steps_per_epoch=count_batches,
        epochs=EPOCHES,
        verbose=1,
        max_queue_size=100,
        use_multiprocessing=True,
        workers=cpu_count(),
        callbacks=[
            ModelCheckpoint(
                os.path.join(WEIGHTS_DIR, 'Epoch_{epoch:02d}___Loss_{loss:.4f}___Acc_{acc:.4f}.hdf5'),
                monitor='acc',
                verbose=1,
                save_weights_only=True,
                save_best_only=True)
        ])
    return history


def save_model(results_path, model):
    """
    Save the model and its weights to files
    @param model: model for saving
    """
    open(os.path.join(results_path, 'MobileNetV1.json'), 'w').write(model.to_json()) # save the architecture to json
    model.save_weights(os.path.join(results_path, 'MobileNetV1.h5'), overwrite=True) # save weights of network
    print('\nModel saved!\n')


def create_folder_results():
    """
    Create the folder for saving results of training the network
    @return: path to folder
    """
    folder_name = datetime.now().strftime('___%d.%m.%Y___%H.%M.%S')
    results_path = os.path.join('../results', folder_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


def plot_results(results_path, history):
    """
    Plot results of training
    @param history: history of training
    """
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
    # history of loss
    plt.clf() # clear the previous shown history
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
    """
    Launch the training of network
    @param gpus: GPU devices, on which the training will be implemented
    @param data_path: path to selection
    @param train_list_file: file contained train data and its labels
    @param val_list_file: file contained validation data and its labels
    """
    optimizer = get_optimizer()
    model = get_model(optimizer, gpus)
    history = train(model, data_path, train_list_file, val_list_file)
    results_path = create_folder_results()
    save_model(results_path, model)
    #plot_results(results_path, history.history)


def init_argparse():
    """
    Initialize argparser
    @return: parsed command-line arguments of cript
    """
    parser = ArgumentParser(description='MobileNet v.1 network')
    parser.add_argument(
        '-gpus',
        '--count_gpus',
        nargs='?',
        help='count of using GPUs',
        default=1,
        type=int)
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
    """
    Main function
    """
    args = init_argparse().parse_args()
    gpus = args.count_gpus
    data_path = args.data_path
    train_list_file = args.train_list
    val_list_file = args.validation_list
    launch_network(gpus, data_path, train_list_file, val_list_file)


if __name__ == '__main__':
    main()
