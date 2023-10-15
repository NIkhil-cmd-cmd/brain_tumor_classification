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

streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Publico', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
training_path = 'data/Training/'
testing_path = 'data/Testing/'
classes = {'notumor': 0, 'pituitary': 1, 'meningioma': 2, 'glioma': 3}
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

st.header('About Brain Tumors')
st.markdown("""
    Brain tumors are abnormal growths of cells in the brain. They can be benign (non-cancerous) or malignant (cancerous), and they can vary in size and location.
    Here are some key facts about brain tumors:
    - Brain tumors can cause a wide range of symptoms, including headaches, seizures, and changes in behavior or cognition.
    - Early detection and diagnosis are crucial for effective treatment and improved outcomes.
    - MRI imaging is a common method for diagnosing brain tumors, and machine learning models can assist in the detection process.
""")
# Add information and example images for each type of tumor
st.header('Types of Brain Tumors')
st.markdown("""
    There are several types of brain tumors, each with its own characteristics. Here are some examples of different brain tumor types:
    - **No Tumor**: This is a normal MRI scan without any tumors.
    - **Pituitary Tumor**: A tumor that develops in the pituitary gland, which can affect hormone regulation.
    - **Meningioma Tumor**: A tumor that arises from the membranes that cover the brain and spinal cord.
    - **Glioma Tumor**: A common type of brain tumor that originates in the glial cells.
""")
st.header('Data Orientations')
st.markdown("""
    The brain tumor data used in this project was taken in three different orientations:
    - Sagittal: Side
    - Coronal: Front
    - Transverse: Top
""")

