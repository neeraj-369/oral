import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('t3_5_epoch.h5')

# List of class names
train_ds = ['all_benign', 'all_early', 'all_pre', 'all_pro', 'brain_glioma', 'brain_menin', 'brain_tumor',
            'breast_benign', 'breast_malignant', 'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi',
            'colon_aca', 'colon_bnt', 'kidney_normal', 'kidney_tumor', 'lung_aca', 'lung_bnt', 'lung_scc', 'lymph_cll',
            'lymph_fl', 'lymph_mcl', 'oral_normal', 'oral_scc']

# find the index of the maximum element in an array
def find_max(arr):
    return np.argmax(arr)

# predict the class from an input image using the trained model
def predict_image_class(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    classes = model.predict(img)
    max_ind = find_max(classes[0])
    return train_ds[max_ind]

# Input image path
input_image_path = './oral_normal_0001.png'  # Replace with the path to your image

# Make a prediction using the input image
predicted_class = predict_image_class(model, input_image_path)

# Display the prediction
print(f"Prediction: {predicted_class}")


