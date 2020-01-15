import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras import backend as K # backend of Keras
from keras.utils import multi_gpu_model
from multiprocessing import cpu_count
from keras.optimizers import TFOptimizer
from keras.callbacks import ModelCheckpoint
import os


K.set_image_dim_ordering('tf')
TRAIN_DIR = '../../../CLS-LOC_multiscale/train'
WEIGHTS_DIR = '../weights'


# parameters of optimizer
LEARNING_RATE = 0.01
DECAY = 0.9
MOMENTUM = 0.9


class_names = [el for el in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, el))]
print('Classes: {}\n'.format(len(class_names)))
datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last',
    fill_mode='nearest')
train_generator = datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(224, 224),
    color_mode='grayscale',
    classes = class_names,
    class_mode='categorical',
    batch_size=256,
    shuffle=True)
with K.tf.device('/cpu:0'):
    kernel_model = MobileNet(
        input_shape=(224, 224, 1),
        alpha=1.0,
        include_top=True,
        weights=None,
        classes=len(class_names))
parallel_model = multi_gpu_model(kernel_model, gpus=8)
optimizer = TFOptimizer(tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,
                                                  decay=DECAY,
                                                  momentum=0.9)) # Tensorflow RMSProp
parallel_model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', # for multiclass classification
    metrics=['accuracy'])
parallel_model.fit_generator(
    train_generator,
    steps_per_epoch=25000,
    epochs=10,
    verbose=1,
    max_queue_size=100,
	use_multiprocessing=True,
    workers=cpu_count(),
    callbacks=[
        ModelCheckpoint(
            os.path.join(WEIGHTS_DIR, 'Epoch_{epoch:03d}___Loss_{loss:.4f}___Acc_{acc:.4f}.hdf5'),
            monitor='acc',
            verbose=1,
            save_weights_only=True,
            save_best_only=True)
    ]
)
open('MobileNetV1.json', 'w').write(parallel_model.to_json()) # save the architecture to json
parallel_model.save_weights('MobileNetV1.h5', overwrite=True) # save weights of network