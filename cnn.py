import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator

data_dir = 'data'
classes = ['DUM555', 'DUM560', 'DUM562', 'DUM587', 'DUM588']

# Return VGG-19 model
def get_model(input_shape):

    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    # print(conv_base.summary())
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    m = models.Sequential()
    m.add(conv_base)
    m.add(layers.Flatten())
    # m.add(layers.Dense(4096, activation='relu'))
    m.add(layers.Dense(256, activation='relu'))
    m.add(layers.Dense(5, activation='softmax'))
    return m

# Self-defined MAPE metrics
import keras.backend as K
def mof_mape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), 1, None))
    return 100. * K.mean(diff, axis=-1)

# One step of k-fold validation
def onefold(n):
    print('Training on fold: {} ...'.format(n))
    model = get_model((240, 320, 3))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=5e-6), metrics=['acc', mof_mape, 'mse'])
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       rotation_range=30,
                                       fill_mode='nearest'
                                       )

    val_datagen = ImageDataGenerator(rescale=1/255.)

    train_generator = train_datagen.flow_from_directory(os.path.join(data_dir, 'train'),
                                                        target_size=(240, 320),
                                                        batch_size=20,
                                                        class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(os.path.join(data_dir, 'validation'),
                                                    target_size=(240, 320),
                                                    batch_size=20,
                                                    class_mode='categorical')
    
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=22,
                                  epochs=150,
                                  validation_data=val_generator,
                                  validation_steps=5,
                                  verbose=2)
    
    model.save('k-fold-04-14-'+str(n)+'.h5')
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b+', label='Training acc')
    plt.plot(epochs, val_acc, 'r+', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b+', label='Training loss')
    plt.plot(epochs, val_loss, 'r+', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    return history

# Preprocess images: crop info bars and resize images
def crop_n_resize(f):

    img = cv2.imread(f)
    if img.shape == (1886, 2048, 3):
        img = img[:1886-118, :]    # (1768, 2048, 3)
        img0 = img[:960, :1280]
        img1 = img[:960, 768:]
        img2 = img[808:, :1280]
        img3 = img[808:, 768:]
        imgs = [img0, img1, img2, img3]
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], (320, 240))
        return imgs
    elif img.shape == (1024, 1280, 3):
        img = img[:1024-64, :]    # (960, 1280, 3)
        img = cv2.resize(img, (320, 240))    # (240, 320, 3)
        return [img]
    else:
        raise Exception('Image of unsupported shape fed.')

# Preprocess images and get ready for ImageGenerator
def copy_file_with_preprocess(filename, src_path, dst_path):

    imgs = crop_n_resize(os.path.join(src_path, filename))
    if len(imgs) == 1:
        cv2.imwrite(os.path.join(dst_path, filename), imgs[0])
    for i in range(len(imgs)):
        cv2.imwrite(os.path.join(dst_path, filename[:-4] + '_' + str(i) + '.tif'), imgs[i])

# k-fold validation
import random
def kfold(k=5):
    from shutil import copyfile, rmtree
    ls_data = os.listdir(data_dir)
    files = {}
    for c in classes:
        tmp = os.listdir(os.path.join(data_dir, c))
        tmp = [x for x in tmp if not x.startswith('.')]
        random.shuffle(tmp)
        files[c] = tmp
    for i in range(1, k+1):
        print('\n----------------\n')
        print('Getting ready for fold: {}'.format(i))
        if 'train' in ls_data:
            rmtree(os.path.join(data_dir, 'train'))
        if 'validation' in ls_data:
            rmtree(os.path.join(data_dir, 'validation'))
        os.mkdir(os.path.join(data_dir, 'train'))
        os.mkdir(os.path.join(data_dir, 'validation'))
        for c in classes:
            os.mkdir(os.path.join(data_dir, 'train', c))
            os.mkdir(os.path.join(data_dir, 'validation', c))
            tmp = files[c]
            size = int(len(tmp) / k)
            tmp_validation = tmp[size * (i-1) : size * i]
            tmp_train = tmp[0 : size * (i-1)] + tmp[size * i :]
            print('{} imgs in class {}'.format(len(tmp), c))
            print('\t{} imgs for training'.format(len(tmp_train)))
            print('\t{} imgs for validation'.format(len(tmp_validation)))
            for t in tmp_train:
                # copyfile(os.path.join(data_dir, c, t), os.path.join(data_dir, 'train', c, t))
                # copy_file_with_preprocess(t, os.path.join(data_dir, c), os.path.join(data_dir, 'train', c))
                copy_file_with_preprocess(t, os.path.join('originals', c), os.path.join(data_dir, 'train', c))
            for v in tmp_validation:
                # copyfile(os.path.join(data_dir, c, v), os.path.join(data_dir, 'validation', c, v))
                # copy_file_with_preprocess(v, os.path.join(data_dir, c), os.path.join(data_dir, 'validation', c))
                copy_file_with_preprocess(v, os.path.join('originals', c), os.path.join(data_dir, 'validation', c))
        return onefold(i)

import time

start_time = time.time()
history = kfold()
print('\n----------------\n')
print('Done. Time elapsed: {:.2f} minutes.'.format((time.time() - start_time) / 60))

# Plot accuracy and loss plots
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b+', label='Training acc')
plt.plot(epochs, val_acc, 'r+', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b+', label='Training loss')
plt.plot(epochs, val_loss, 'r+', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Plot MAPE and MSE plots
mape = history.history['mof_mape']
val_mape = history.history['val_mof_mape']
mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, mape, 'b+', label='Training MAPE')
plt.plot(epochs, val_mape, 'r+', label='Validation MAPE')
plt.title('Training and validation MAPE')
plt.legend()
plt.figure()
plt.plot(epochs, mse, 'b+', label='Training MSE')
plt.plot(epochs, val_mse, 'r+', label='Validation MSE')
plt.title('Training and validation MSE')
plt.legend()
plt.show()

fig, ax1 = plt.subplots(figsize=(20,10))
color = 'tab:red'
ax1.set_xlabel('epochs', fontsize=15)
ax1.set_ylabel('Training MAPE', color=color, fontsize=15)
ax1.plot(epochs, mape, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Training MSE', color=color, fontsize=15)
ax2.plot(epochs, mse, color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.savefig('mape-mse.png', dpi=300)
plt.show()

