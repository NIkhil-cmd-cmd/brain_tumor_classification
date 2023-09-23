import numpy as np
import cv2
import os
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import base64 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def set_bg_hack(main_bg):
    main_bg_ext = "png"
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack('magicpattern-mesh-gradient-1695322086991.png')

st.set_option('deprecation.showPyplotGlobalUse', False)

training_path = 'data/Training'
testing_path = 'data/Testing'

classes = {'no_tumor': 0, 'pituitary_tumor': 1, 'meningioma_tumor': 2, 'glioma_tumor': 3}

X_train, Y_train, X_test, Y_test = [], [], [], []

# Load training data
for cls in classes:
    pth = os.path.join(training_path, cls)
    for j in os.listdir(pth):
        img = cv2.imread(os.path.join(pth, j), 0)
        img = cv2.resize(img, (200, 200))
        X_train.append(img)
        Y_train.append(classes[cls])

# Load testing data
for cls in classes:
    pth = os.path.join(testing_path, cls)
    for j in os.listdir(pth):
        img = cv2.imread(os.path.join(pth, j), 0)
        img = cv2.resize(img, (200, 200))
        X_test.append(img)
        Y_test.append(classes[cls])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# PCA
n_components = 50
pca = PCA(n_components=n_components)
pca.fit(X_train)
pca_train = pca.transform(X_train)
pca_test = pca.transform(X_test)

# Train classifier
sv = SVC()
sv.fit(pca_train, Y_train)

# UI
st.title('Brain Tumor Detection through MRI Imaging')
st.sidebar.header('User Input')
uploaded_image = st.sidebar.file_uploader("Upload an image for prediction", type=["jpg", "png"])

if uploaded_image is not None:
    st.sidebar.image(uploaded_image, use_column_width=True, caption="Uploaded Image")

# Predictions
if st.sidebar.button("Predict"):
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (200, 200))
        img = img.reshape(1, -1) / 255
        img_pca = pca.transform(img)

        prediction = sv.predict(img_pca)
        class_names = {0: 'No Tumor', 1: 'Pituitary Tumor', 2: 'Meningioma Tumor', 3: 'Glioma Tumor'}
        result = class_names[prediction[0]]

        st.write(f"Prediction: {result}")

st.header('Random Test Images')
num_images_to_display = 9
random_indices = np.random.randint(0, len(X_test), num_images_to_display)

for i, index in enumerate(random_indices):
    plt.subplot(3, 3, i + 1)
    image_to_display = X_test[index].reshape(200, 200)
    plt.imshow(image_to_display, cmap='gray', vmin=0, vmax=1)
    actual_class = [k for k, v in classes.items() if v == Y_test[index]][0]
    plt.title(f"Actual: {actual_class}", fontsize=6)
    plt.axis('off')
    img = X_test[index].reshape(1, -1)
    img_pca = pca.transform(img)
    prediction = sv.predict(img_pca)
    class_names = {0: 'No Tumor', 1: 'Pituitary Tumor', 2: 'Meningioma Tumor', 3: 'Glioma Tumor'}
    predicted_class = class_names[prediction[0]]
    color = ('green' if (prediction[0] == Y_test[index]) else 'red')
    plt.text(10, 180, f"Predicted: {predicted_class}", color=color, fontsize=6, bbox=dict(facecolor='white', alpha=0.8))
    

st.pyplot()
st.header('Model Evaluation')
st.write("Testing Score (Accuracy):", accuracy_score(Y_test, sv.predict(pca_test)))

# Confusion Matrix
st.subheader('Accuracy Matrix')
conf_matrix = confusion_matrix(Y_test, sv.predict(pca_test))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names.values(), yticklabels=class_names.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot()

# Classification Report
st.subheader('Classification Report')
classification_rep = classification_report(Y_test, sv.predict(pca_test), target_names=class_names.values())
st.text(classification_rep)


