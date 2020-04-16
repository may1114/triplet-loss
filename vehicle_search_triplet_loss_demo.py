#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.backend as K
MARGIN = 1
def triplet_loss(vecs):
    ancor, pos, neg = vecs
    #l2 normalize those data
    l2_ancor = K.l2_normalize(ancor, axis = -1)
    l2_pos = K.l2_normalize(pos, axis = -1)
    l2_neg = K.l2_normalize(neg, axis = -1)
    
    distance_ancor_pos = K.sum(K.square(K.abs(l2_ancor - l2_pos)), axis = -1, keepdims= True)
    distance_ancor_neg = K.sum(K.square(K.abs(l2_ancor - l2_neg)), axis = -1, keepdims = True)
    
    loss = distance_ancor_pos + MARGIN - distance_ancor_neg
    return loss

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


# In[ ]:


#use triplet identify the vehicle
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStop, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Input, Lambda
from keras.models import Model


LEARNING_RATE = 0.00001
IMG_WIDTH = 299
IMG_HEIGHT = 299
NBR_MODELS = 250
NBR_COLORS = 7
INITIAL_EPOCH = 0
#define the model, we get imagenet weights from InceptionV3, but don't need the top layer
inception = InceptionV3(include_top = False, input_tensor = None, input_shape= (IMG_WIDTH, IMG_HEIGHT, 3), pooling = 'avg')
f_base = inception.get_layers(index = -1).output

f_acs = Dense(1024, name='f_acs')(f_base)
feature_model = Model(inputs = inception.input, outputs = f_acs)

ancor = Input(input_shape= (IMG_WIDTH, IMG_HEIGHT, 3), name='ancor')
positive = Input(input_shape= (IMG_WIDTH, IMG_HEIGHT, 3), name='positive')
negative = Input(input_shape= (IMG_WIDTH, IMG_HEIGHT, 3), name='negative')

#the function of classify the car model and color is only for the ancor
#now get the ancor f_acs layers feature
#after this layer, will do model classify and color classify
f_acs_ancor = feature_model(ancor)
f_ancor_model = Dense(NBR_MODELS, activation='softmax', name='pred_model')(f_acs_ancor)
f_ancor_color = Dense(NBR_COLORS, activation='softmax', name = 'pred_color')(f_acs_ancor)

#now create the triplet branch, to check the similarity with the positive and negative
f_sls1 = Dense(1024, name='f_sls1')(f_base)
#for the seconde layer of this branch, we need to concate the data from the f_acs layer
f_sls2_data = concatenate([f_acs_ancor, f_sls1], axis = -1)
f_sls2 = Dense(1024, name='f_sls2')(f_sls2_data)
#build third layer 
f_sls3 = Dense(256, name='f_sls3')(f_sls2)
#this model is just for one picture, try to get it's embedding data
sls_model = Model(inputs = inception.input, outputs=f_sls3)
#ancor embedding data
sls_ancor = sls_model(ancor)
sls_positive = sls_model(positive)
sls_neg = sls_model(negative)
#then after get those 3 image's embedding data, we can compute the loss
loss = Lambda(triplet_loss, shape=(1, ))([sls_ancor, sls_positive, sls_neg])

#after build the classify model branch, similarity branch, now we build the whole model
#the model inputs is those 3 image, and outputs is the classify result and loss
model = Model(inputs=[ancor, positive, negative], outputs=[f_ancor_model, f_ancor_color, loss])

#create optimizer
optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)
#compile model
model.compile(loss=["categorical_crossentropy", "categorical_crossentropy", identity_loss], optimizer=optimizer, metrics=["accuracy"])
#model sumary
model.sumary()
#create callbacks
save_model = ModelCheckpoint('./models/inception3_vehicle_modelcolor_triplet.h5', verbose = 1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience = 3, min_lr= 0.00001)
early_stop = EarlyStop(monitor='val_loss', patience= 15)
#get data


#fit generator

