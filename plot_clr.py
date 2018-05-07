import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import SGD, Adam

from clr import OneCycleLR

plt.style.use('seaborn-white')

# Constants
NUM_SAMPLES = 2000
NUM_EPOCHS = 100
BATCH_SIZE = 500
INITIAL_LR = 0.1

# Data
X = np.random.rand(NUM_SAMPLES, 10)
Y = np.random.randint(0, 2, size=NUM_SAMPLES)

# Model
inp = Input(shape=(10,))
x = Dense(5, activation='relu')(inp)
x = Dense(1, activation='sigmoid')(x)
model = Model(inp, x)

clr_triangular = OneCycleLR(NUM_SAMPLES, NUM_EPOCHS, BATCH_SIZE, INITIAL_LR,
                            end_percentage=0.1, scale_percentage=None)

model.compile(optimizer=SGD(0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[clr_triangular], verbose=0)


plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("CLR - 'triangular' Policy")
plt.plot(clr_triangular.history['lr'])
plt.show()

plt.xlabel('Training Iterations')
plt.ylabel('Momentum')
plt.title("CLR - 'triangular' Policy")
plt.plot(clr_triangular.history['momentum'])
plt.show()
