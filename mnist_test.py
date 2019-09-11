import tensorflow as tf
from tensorflow import keras

from PD_optimizer import PD
from PID_optimizer import PID

import numpy as np
import matplotlib.pyplot as plt

learning_epochs = 100
batch_size = 100
method = 'adam'   # select 'sgd', 'sgd-momentum', 'PD', 'PID', 'adam'

# Optimization method selection
if method == 'sgd-momentum':
    optimizer = keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=False)
elif method == 'PD':
    optimizer = PD(lr=0.2, kd=0.05)
elif method == 'PID':
    optimizer = PID(lr=0.01, momentum=0.9, kd=0.5)
else:
    optimizer = method
print('Optimization method = {}'.format(method))

# fix random seed (if necessary)
tf.random.set_seed(42)
np.random.seed(42)

# load mnist dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# network definition
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# learning and evaluation
epochs = range(1, learning_epochs+1)
train_accs = []
test_accs = []
model.compile(optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
for epoch in epochs:
    history = model.fit(train_images, train_labels, 
            batch_size=batch_size,
            epochs=1)
    train_accs.append(history.history['accuracy'][0])
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test_accs.append(test_acc)

plt.figure()
plt.plot(epochs, train_accs, label='train')
plt.plot(epochs, test_accs, label='test')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid(True)
plt.show()

# save learning data
with open('{}_result.csv'.format(method), 'w') as f:
    for epoch, train_acc, test_acc in zip(epochs, train_accs, test_accs):
        f.write('{0:d},{1},{2}\n'.format(epoch, train_acc, test_acc))

