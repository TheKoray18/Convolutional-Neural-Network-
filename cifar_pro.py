# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 22:14:57 2020

@author: Koray
"""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout,Activation
from keras.utils import to_categorical

#%% Datamızı Train-Test olarak ayırıyoruz

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#%% Datalarımızın Boyutuna Bakalım

print("x_train :", x_train.shape)
print("y_train :", y_train.shape)
print("x_test :", x_test.shape)
print("y_test :", y_test.shape)

#%% Train ve Test Sayısı 

print(x_train.shape[0],"train örnekleri")
print(x_test.shape[0],"test örnekleri")

#%% x_train ve x_test datalarımızı tipimi uint8'den float'a çeviriyoruz

x_train=x_train.astype('float32')

x_test=x_test.astype('float32')

#%% x_train ve x_test datalarımızı 0-1 yapıyoruz

x_train=x_train/255

x_test=x_test/255

#%% x_train içerisinden bir örneğe baktık

import matplotlib.pyplot as plt

plt.imshow(x_train[10])
plt.axis('off')

#%% y_train ve y_test dataları binary forma dönüştürülür
#Verimizde 10 tana sınıf vardır

num_classes=10

y_train=to_categorical(y_train,num_classes=num_classes)
y_test=to_categorical(y_test,num_classes=num_classes)

#%% Sinir ağımızda ilk layerımızda ki input shape için 

input_shape=x_train.shape[1:]

print(input_shape)

#"(32,32,3)"input shape oluşturduk

#%% CNN Oluşturuyoruz

#Activasyon Fonksiyonu olarak 'Relu' kullanıyoruz
#Overfitting'i engellemek için de Dropout kullanıyoruz

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='same',input_shape=input_shape))
model.add(Activation(activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),padding='same'))
model.add(Activation(activation='relu'))

model.add(Conv2D(64,kernel_size=(3,3),padding='same'))
model.add(Activation(activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))

model.add(Conv2D(64,kernel_size=(3,3),padding='same'))
model.add(Activation(activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.25))

#Multiclass Classifier olduğu için son katmanımızda activation fonksiyonumuz 'Softmax'
model.add(Dense(num_classes,activation='softmax'))

#Modelimizi Compile Ediyoruz 

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#Modelimizi Fit Ediyoruz(Eğitim)

history=model.fit(x_train,y_train,epochs=10,batch_size=50,validation_data=(x_test,y_test))

#%% Modeli Görselleştirme 

import matplotlib.pyplot as plt

training_acc=history.history['accuracy']
test_acc=history.history['val_accuracy']

plt.plot(training_acc,'r--')
plt.plot(test_acc,'b-')
plt.legend(['Training acc','Test acc'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


training_loss=history.history['loss']
test_loss=history.history['val_loss']

plt.plot(training_loss,'r--')
plt.plot(test_loss,'b-')
plt.legend(['training loss','Test loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()




 




























