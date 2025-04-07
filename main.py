import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import streamlit as st
import hashlib
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import time
import requests
from streamlit_lottie import st_lottie
import json
import altair as alt
import pandas as pd
import io
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np

# Define the ResNet model architecture
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the number of classes
num_classes = 7

# Initialize the model
model = ResNet50(num_classes=num_classes)

# Check if the model file exists
model_file = 'resnet50_skin_classifier.pth'
if not os.path.isfile(model_file):
    st.error(f"Model file '{model_file}' not found. Please ensure the model file is in the correct directory.")
else:
    # Load the state dictionary
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

# Define image preprocessing function with enhancement
def enhance_image(image, contrast, sharpness):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    return image

def preprocess_image(image, contrast=1.0, sharpness=1.0):
    image = enhance_image(image, contrast, sharpness)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define the label map
label_map = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Backup old history data
c.execute('''CREATE TABLE IF NOT EXISTS history_backup AS SELECT * FROM history''')
conn.commit()

# Recreate history table
c.execute('''DROP TABLE IF EXISTS history''')
c.execute('''CREATE TABLE history
             (username TEXT, image_path TEXT, prediction TEXT, confidence REAL, date TEXT)''')

# Restore data from backup
c.execute('''INSERT INTO history SELECT * FROM history_backup''')
c.execute('''DROP TABLE history_backup''')
conn.commit()

c.execute('''CREATE TABLE IF NOT EXISTS users
             (username TEXT PRIMARY KEY, password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS symptoms
             (username TEXT, symptom TEXT, severity INTEGER, date TEXT)''')
conn.commit()

# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate_user(username, password):
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    return c.fetchone() is not None

def reset_password(username, new_password):
    hashed_password = hash_password(new_password)
    c.execute("UPDATE users SET password=? WHERE username=?", (hashed_password, username))
    conn.commit()
    return c.rowcount > 0

def get_usernames():
    c.execute("SELECT username FROM users")
    return [row[0] for row in c.fetchall()]

def save_history(username, image_path, prediction, confidence):
    date = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?)", (username, image_path, prediction, confidence, date))
    conn.commit()

def get_history(username):
    c.execute("SELECT * FROM history WHERE username=?", (username,))
    return c.fetchall()

def save_symptom(username, symptom, severity):
    date = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO symptoms VALUES (?, ?, ?, ?)", (username, symptom, severity, date))
    conn.commit()

def get_symptoms(username):
    c.execute("SELECT * FROM symptoms WHERE username=?", (username,))
    return c.fetchall()

# Lottie animation loader
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Card layout
def create_card(title, content):
    st.markdown(f"""
    <div style="
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 20px;">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# Image segmentation function (placeholder)
def segment_image(image):
    # Implement actual segmentation logic here
    return image  # For now, just return the original image

# 3D model visualization function
def visualize_3d_model(image):
    z_data = np.asarray(image.convert('L')).astype(np.float64)  # Convert image to grayscale
    fig = go.Figure(data=[go.Surface(z=z_data)])
    fig.update_layout(title='3D Model of Skin Lesion', autosize=True,
                      width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

# PDF report generation
def create_pdf_report(results):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 750, "Skin Lesion Classification Report")
    y = 700
    for result in results:
        p.drawString(100, y, f"Image: {result['image']}")
        p.drawString(100, y - 20, f"Prediction: {result['prediction']}")
        p.drawString(100, y - 40, f"Confidence: {result['confidence']:.2f}")
        y -= 60
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# Time series analysis
def plot_lesion_changes(history):
    dates = [entry[4] for entry in history]  # Assuming date is the 5th item in history tuple
    confidences = [entry[3] for entry in history]  # Assuming confidence is the 4th item
    fig, ax = plt.subplots()
    ax.plot(dates, confidences)
    ax.set_title("Lesion Classification Confidence Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Confidence")
    plt.xticks(rotation=45)
    return fig

# Risk assessment function
def calculate_risk_score(user_data, symptoms):
    # Implement actual risk calculation logic here
    risk_score = 0.5  # Placeholder risk score
    if symptoms:
        total_severity = sum([symptom[2] for symptom in symptoms])
        risk_score += total_severity * 0.1
    if len(user_data) > 0:
        avg_confidence = sum([entry[3] for entry in user_data]) / len(user_data)
        risk_score += avg_confidence * 0.2
    return min(risk_score, 1.0)  # Ensure risk score does not exceed 1.0

def main():
    # Set up Streamlit app
    st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")

    # Sidebar for navigation and settings
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload", "History", "Symptoms", "About"])

    # Theme selection
    theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
            <style>
            body {
                color: #fff;
                background-color: #1E1E1E;
            }
            </style>
            """, unsafe_allow_html=True)

    # Main app logic
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'reset_password' not in st.session_state:
        st.session_state.reset_password = False

    # Login/Register logic
    if not st.session_state.logged_in and not st.session_state.reset_password:
        st.title("🌟 Skin Lesion Classifier 🌟")

        # Lottie animation
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_UJNc2t.json"
        lottie_json = load_lottieurl(lottie_url)
        st_lottie(lottie_json, speed=1, height=200, key="initial")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully.")
            else:
                st.error("Invalid username or password.")
        if st.button("Register"):
            if register_user(username, password):
                st.success("Registration successful. You can now log in.")
            else:
                st.error("Username already exists. Please choose a different one.")
        if st.button("Forgot Password"):
            st.session_state.reset_password = True

    elif st.session_state.reset_password:
        st.title("🌟 Reset Password 🌟")
        username = st.text_input("Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Reset Password"):
            if username in get_usernames():
                if reset_password(username, new_password):
                    st.success("Password reset successfully. You can now log in.")
                    st.session_state.reset_password = False
                else:
                    st.error("Failed to reset password. Please try again.")
            else:
                st.error("Username does not exist.")
        if st.button("Back to Login"):
            st.session_state.reset_password = False

    else:
        st.title(f"🌟 Skin Lesion Classifier 🌟 - Welcome, {st.session_state.username}!")

        if page == "Home":
            create_card("Welcome", "Upload your skin lesion image for classification.")

            # Tutorial slides
            tutorial_slides = [
                {"title": "Welcome", "content": "Let's get started with skin lesion classification!"},
                {"title": "Step 1", "content": "Upload your image"},
                {"title": "Step 2", "content": "Adjust enhancement settings if needed"},
                {"title": "Step 3", "content": "View your results"},
            ]

            current_slide = st.empty()
            for slide in tutorial_slides:
                with current_slide.container():
                    st.subheader(slide["title"])
                    st.write(slide["content"])
                    time.sleep(3)  # Each slide shows for 3 seconds

        elif page == "Upload":
            st.header("Upload an image to classify")

            # Drag and drop upload
            st.markdown("""
                <style>
                .upload-area {
                    border: 2px dashed #ccc;
                    border-radius: 20px;
                    padding: 20px;
                    text-align: center;
                }
                </style>
                <div class="upload-area" ondrop="drop(event)" ondragover="allowDrop(event)">
                    <p>Drag and drop your image here or click to upload</p>
                </div>
                <script>
                function allowDrop(ev) {
                    ev.preventDefault();
                }
                function drop(ev) {
                    ev.preventDefault();
                    var files = ev.dataTransfer.files;
                    document.getElementById('file_uploader').files = files;
                }
                </script>
            """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader("Select image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

            if uploaded_files:
                results = []
                for uploaded_file in uploaded_files:
                    original_image = Image.open(uploaded_file).convert('RGB')

                    # Image comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_image, caption="Original Image")

                    enhance = st.checkbox("Enhance image before classification")
                    if enhance:
                        contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
                        sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0)
                        enhanced_image = enhance_image(original_image, contrast, sharpness)
                        with col2:
                            st.image(enhanced_image, caption="Enhanced Image")
                    else:
                        contrast, sharpness = 1.0, 1.0

                    if st.button(f"Classify {uploaded_file.name}"):
                        progress_bar = st.progress(0)
                        with st.spinner("Image is being processed..."):
                            for percent_complete in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(percent_complete + 1)

                            processed_image = preprocess_image(original_image, contrast, sharpness).to(device)
                            with torch.no_grad():
                                output = model(processed_image)
                                prediction_confidence, prediction_class = torch.max(output, 1)
                                prediction_confidence = torch.nn.functional.softmax(output, dim=1)[0][
                                    prediction_class].item()

                            predicted_label = label_map[prediction_class.item()]
                            st.markdown(f"### Predicted label: **{predicted_label}**")
                            st.markdown(f"### Confidence: **{prediction_confidence:.2f}**")

                            # Visualization
                            fig = go.Figure(go.Bar(
                                x=[prediction_confidence],
                                y=[predicted_label],
                                orientation='h'))
                            st.plotly_chart(fig)

                            # Save to history
                            save_history(st.session_state.username, uploaded_file.name, predicted_label,
                                         prediction_confidence)

                            results.append({
                                'image': uploaded_file.name,
                                'prediction': predicted_label,
                                'confidence': prediction_confidence
                            })

                            # 3D Model Visualization
                            st.subheader("3D Model Visualization")
                            visualize_3d_model(original_image)

                            # Image segmentation
                            segmented_image = segment_image(original_image)
                            st.image(segmented_image, caption="Segmented Lesion")

                            # User feedback
                            feedback = st.radio("Was this classification helpful?", ("Yes", "No"))
                            if feedback == "No":
                                st.text_area("Please tell us why:")

                # Export report
                if results and st.button("Export Report"):
                    pdf = create_pdf_report(results)
                    st.download_button("Download PDF Report", pdf, "skin_lesion_report.pdf")

        elif page == "History":
            st.header("Your classification history")
            history = get_history(st.session_state.username)
            for entry in history:
                st.write(f"Image: {entry[1]}, Prediction: {entry[2]}, Confidence: {entry[3]:.2f}, Date: {entry[4]}")

            # Data visualization dashboard
            history_data = pd.DataFrame(history, columns=['username', 'image', 'prediction', 'confidence', 'date'])
            chart = alt.Chart(history_data).mark_bar().encode(
                x='prediction:N',
                y='count():Q',
                color='prediction:N'
            )
            st.altair_chart(chart, use_container_width=True)

            # Time series analysis
            if len(history) > 1:
                st.subheader("Lesion Classification Confidence Over Time")
                fig = plot_lesion_changes(history)
                st.pyplot(fig)

            # Risk assessment
            st.subheader("Personal Risk Assessment")
            symptoms = get_symptoms(st.session_state.username)
            risk_score = calculate_risk_score(history, symptoms)
            st.write(f"Your personal risk score: {risk_score:.2f}")
            st.write(
                "Note: This is a simplified risk assessment. Please consult a healthcare professional for accurate medical advice.")

        elif page == "Symptoms":
            st.header("Track Your Symptoms")
            symptom = st.text_input("Symptom")
            severity = st.slider("Severity", 1, 10)
            if st.button("Add Symptom"):
                save_symptom(st.session_state.username, symptom, severity)
                st.success("Symptom added successfully.")

            st.subheader("Your Symptoms")
            symptoms = get_symptoms(st.session_state.username)
            for symptom in symptoms:
                st.write(f"Symptom: {symptom[1]}, Severity: {symptom[2]}, Date: {symptom[3]}")

        elif page == "About":
            st.header("About Skin Lesion Classification")
            st.write("How to use this system:")
            st.write("1. Navigate to the Upload page")
            st.write("2. Upload an image of a skin lesion")
            st.write("3. Click 'Classify' to get the classification result")
            st.write("4. View your history in the History page")
            st.write("5. Track your symptoms in the Symptoms page")

            # Educational resources
            st.subheader("Learn about skin lesions")
            lesion_types = {
                "Melanoma": "Melanoma is a serious form of skin cancer that begins in cells known as melanocytes.",
                "Basal Cell Carcinoma": "Basal cell carcinoma is the most common form of skin cancer and the most frequently occurring form of all cancers.",
                "Squamous Cell Carcinoma": "Squamous cell carcinoma is the second most common form of skin cancer. It's usually found on areas of the body damaged by UV rays from the sun or tanning beds.",
                "Actinic Keratosis": "Actinic keratosis is a rough, scaly patch on the skin that develops from years of sun exposure.",
                "Nevus": "A nevus is a common skin growth that develops when pigment cells in the skin grow in clusters or clumps.",
                "Seborrheic Keratosis": "Seborrheic keratoses are common noncancerous skin growths that often appear as people age.",
            }

            selected_lesion = st.selectbox("Select a lesion type to learn more", list(lesion_types.keys()))
            st.write(lesion_types[selected_lesion])

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.experimental_rerun()

if __name__ == "__main__":
    main()
