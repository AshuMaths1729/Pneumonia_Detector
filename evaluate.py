import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Hyperparams
IMAGE_SIZE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 1
BATCH_SIZE = 16

model = load_model('saved_models/trained_model.h5')
path = 'data/test/NORMAL/IM-0086-0001.jpeg'
test_data_dir = "data/test"
img = cv2.imread(path,-1)
plt.imshow(img)
img = image.load_img(path, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#result = []
images = np.vstack([x])
classes = model.predict(images)
print(classes[0])
if classes[0]>0.5:
    print("Predicted PNEUMONIA")
else:
    print("Predicted NORMAL")