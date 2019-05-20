import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

class_dir = './data/examples/captainmarvel.jpg'
classes = ['cat', 'dog']

model2 = load_model('./models/inceptionv3-fine-tune.model')
model2.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

img_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

original = cv2.imread(class_dir)
preprocessed = img_gen.standardize(original)
formatted = np.expand_dims(preprocessed, 0)

prediction = model2.predict(formatted)
print(prediction)
prediction = np.argmax(prediction)


plt.figure()
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.xlabel("I think this is a " + classes[prediction])
plt.savefig("./catdog/CatOrDog.jpg")
