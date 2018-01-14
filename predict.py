from keras.models import load_model
import cv2
import numpy as np

model = load_model('classifier.h5')

img = cv2.imread('image0048.jpg')
img = cv2.resize(img,(28,28))
img = np.reshape(img,[1,28,28,3])

classes = model.predict(img)

# predict_classes, proba
print(classes)
