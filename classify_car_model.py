#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import ceil
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from sklearn.utils import class_weight
from utils import generator_batch, generator_batch_multitask

FINE_TUNE = True
USE_PROCESSING =False
NEW_OPTIMIZER = True
LEARNING_RATE = 0.001
NBR_EPOCHS = 100
BATCH_SIZE = 64
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'acc'
NBR_MODELS = 250
USE_CLASS_WEIGHTS = False
RANDOM_SCALE = True
MULTI_TASK = False
NBR_COLOR = 7

train_path = './train_vehicleModel_list.txt'
val_path = './val_vehicleModel_list.txt'


# In[ ]:


from keras.models import load_model

if FINE_TUNE:
    #can load already fine_tune trained model
    if MULTI_TASK:
        model = load_model('./models/inception3_fine_tune_multi_task.h5')
    else:
        model = load_model('./models/inception3_fine_tune.h5')
else:
    inception_model = InceptionV3(include_top = False, wegihts = None, input_tensor = None, input_shape(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')
    x = inception_model.get_layer(index = -1).output
    x = Dense(1024, activation='relu')(x)
    model_out = Dense(NBR_MODELS, activation='softmax')(x)
    if MULTI_TASK:
        color_out = = Dense(NBR_COLOR, activation='softmax')(x)
    
    model = Model(inputs=inception_model.input, outputs = [model_out, color_out])
    
    #after create model, create optimizer
    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, nesterov = True)
    #compile model
    save_file = './models/inceptionv3_category_model.h5'
    if MULTI_TASK:
        save_file = './models/inceptionv3_model_color.h5'
        model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  #loss_weights = [0.6, 0.4],
                  optimizer=optimizer, metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    #create check point, early stop, change learning rate
    best_model = ModelCheckpoint('./models/inceptionv3_category_model.h5', monitor='val_'+monitor_index,
                                verbose = 1, save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor = 0.5, patience = 3, verbose = 1, min_lr = 0.00001)
    early_stop = EarlyStop(monitor='val_'+monitor_index, patience=15, verbose = 1)
    
    train_data, train_label, val_data = get_train_val_data()
    
    steps_per_epoch = int(ceil(len(train_data) * 1. / BATCH_SIZE))
    validation_steps = int(ceil(len(val_data) * 1. / BATCH_SIZE))
    if USE_CLASS_WEIGHTS:
        # balance the wegiht
        class_weights = class_weight('balanced', np.unique(train_label), train_label)
    
    #train model
    if MULTI_TASK:
        model.fit_generator(generator_batch_multitask(train_data_lines,
                        nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch_multitask(val_data_lines,
                        nbr_class_one = NBR_MODELS, nbr_class_two = NBR_COLORS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        class_weight = class_weights, callbacks = [best_model, reduce_lr],
                        max_queue_size = 80, workers = 8, use_multiprocessing=True)
    else:
        model.fit_generator(generator_batch(train_data_lines, NBR_MODELS = NBR_MODELS,
                            batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                            img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                            shuffle = True, augment = True),
                            steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                            validation_data = generator_batch(val_data_lines,
                            NBR_MODELS = NBR_MODELS, batch_size = BATCH_SIZE,
                            img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                            shuffle = False, augment = False),
                            validation_steps = validation_steps,
                            class_weight = class_weights, callbacks = [best_model, reduce_lr, early_stop],
                            max_queue_size = 80, workers = 8, use_multiprocessing=True)


# In[ ]:


def get_train_val_data():
    train_data_lines = open(train_path).readlines()
    #only need those existed image files
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    train_labels = [int(w.strip().split(' ')[-1]) for w in train_data_lines]
    
    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    
    return train_data_lines, train_labels, val_data_lines


# In[ ]:


def get_fine_tune_model(train_data_lines, val_data_lines, steps_per_epoch, validation_steps, best_model_file):
    base_model = InceptionV3(include_top = False)
    x = base_model.output
    x = GlobalAveragePooling2D(x)
    x = Dense(1024, activation='relu')(x)
    pred = Dense(NBR_MODELS, activation='softmax')(x)
    
    model = Model(inputs = base_model.input, outputs = pred)
    #first, lock the base model layers weight
    #only train the top layers
    for layer in model.layers:
        layer.trainable = False
        
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc',
                                verbose = 1, save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)
    #compile model
    model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy')
    model.fit_generator(generator_batch(train_data_lines, NBR_MODELS = NBR_MODELS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True), 
                       steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch(val_data_lines,
                        NBR_MODELS = NBR_MODELS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [best_model, reduce_lr],
                        max_queue_size = 80, workers = 8, use_multiprocessing=True)
    
    # after train the top weights, then make from the 250th layer of the base model can train
    for layer in base_model.layers[:249]:
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True
    
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    model.fit_generator(generator_batch(train_data_lines, NBR_MODELS = NBR_MODELS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True), 
                       steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = generator_batch(val_data_lines,
                        NBR_MODELS = NBR_MODELS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [best_model, reduce_lr],
                        max_queue_size = 80, workers = 8, use_multiprocessing=True)

