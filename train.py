import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import scipy
import scipy.misc as sm
from PIL import Image
import csv

cls_names=['anger','disgust','fear','happy','normal','sad','surprised']
train_dir='../my/fer2013/train'
val_dir='../my/fer2013/val'
test_dir='../my/fer2013/test'
h, w, ch=48, 48, 1
bs, n_cls=64, 11

# 数据增强
train_gen=keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, # 将像素值缩放到0和1之间
    rotation_range=20,  # 将像素值缩放到0和1之间
    width_shift_range=0.3,  # 随机水平平移图像的宽度范围（图像宽度的20%）
    height_shift_range=0.3, # 随机水平平移图像的宽度范围（图像宽度的20%）
    shear_range=0.3,    # 随机水平平移图像的宽度范围（图像宽度的20%）
    zoom_range=0.3, # 随机水平平移图像的宽度范围（图像宽度的20%）
    horizontal_flip=True,   # 随机水平平移图像的宽度范围（图像宽度的20%）
    fill_mode='nearest' # 填充新创建的像素的方法（使用最近邻像素的值）
)
train_it=train_gen.flow_from_directory(
    train_dir,
    target_size=(h,w),
    batch_size=bs,
    seed=11,
    shuffle=True,
    class_mode='categorical')

# 验证
val_gen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_it=val_gen.flow_from_directory(
    val_dir,
    target_size=(h,w),
    batch_size=bs,
    seed=11,
    shuffle=False,
    class_mode='categorical'
)

# 测试
test_gen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_it=test_gen.flow_from_directory(
    test_dir,
    target_size=(h,w),
    batch_size=bs,
    seed=11,
    shuffle=False,
    class_mode='categorical'
)

# 获取训练和验证样本数
train_num=train_it.samples
val_num=val_it.samples
print(train_num, val_num)

# 构建模型
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=(w,h,3)))
model.add(keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(n_cls,activation='softmax'))

# 编译模型
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# 训练模型
epochs = 40
history=model.fit_generator(
    train_it,
    steps_per_epoch=train_num//bs,
    epochs=epochs,
    validation_data=val_it,
    validation_steps=val_num//bs
    )
# 保存模型
model.save('model.h5')
