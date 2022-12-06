"""
create a neural network model to predict leaf region
input: 2d ir image
output: 2d mask image. 1 means leaf, 0 means background
"""
import numpy as np
import tensorflow as tf

from src.ginseng_rgb_image import GinsengRgbImage
from src.ir_image import IrImage

WIDTH = 160
HEIGHT = 120

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.summary()

ir = IrImage('../ginseng-ir/irimage_20220512_0800.dat')
rgb = GinsengRgbImage('../ginseng-rgb/camimage_20220512_0800.jpg')

x = ir.image.reshape((-1, HEIGHT, WIDTH))
y = np.array([1])

model.fit(x, y, epochs=15)

predict = model.predict(x[0:1])
