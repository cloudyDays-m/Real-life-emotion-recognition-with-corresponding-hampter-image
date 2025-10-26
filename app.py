import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="emotion detection website",
    layout="wide"
)

st.markdown("""
    <style>
      .main-header {
        font-size: 7rem;
        font-weight: bold;
        text-align: center;
        color: #AAAAEC;

    }
    .emotion-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('my_model_64p35.h5')
        return model
    except Exception as e:
        st.error(f"error loading model: {e}")
        return None

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_data
def load_hampter_images():
    EMO_DIR = Path("hampter-images")
    emotion_names = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]
    file_map = {
        "Angry": "angryh.jpg",
        "Happy": "happyh.jpg",
        "Neutral": "neutralh.jpg",
        "Sad": "sadh.jpg",
        "Surprised": "suprisedh.jpg"
    }
    
    hampter_images = {}
    for emotion in emotion_names:
        filepath = EMO_DIR / file_map[emotion]
        if filepath.exists():
            img = cv2.imread(str(filepath))
            if img is not None:
                hampter_images[emotion] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return hampter_images

def detect_emotion(image, model, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, None, None
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    facess = face_cascade.detectMultiScale(roi_gray)
    
    if len(facess) > 0:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey:ey+eh, ex:ex+ew]
        
        final_image = cv2.resize(face_roi, (224, 224))
        final_image = np.expand_dims(final_image, axis=0)
        final_image = preprocess_input(final_image)
        
        predictions = model.predict(final_image, verbose=0)
        emotion_index = np.argmax(predictions[0])
        
        emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]
        emotion = emotions[emotion_index]
        confidence = predictions[0][emotion_index] * 100
        
        colors = {
            "Angry": (255, 0, 0),
            "Happy": (0, 255, 0),
            "Neutral": (255, 255, 0),
            "Sad": (0, 0, 255),
            "Surprised": (255, 165, 0)
        }
        color = colors[emotion]
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        return emotion, confidence, image
    
    return None, None, None

def main():
    st.markdown('<p class="main-header" style="font-size: 5rem; text-align: center; margin-top: 3rem, margin-bottom: 1rem">Emotion Detection App</p>', unsafe_allow_html=True)
    
    model = load_model()
    face_cascade = load_face_cascade()
    hampter_images = load_hampter_images()
    
    if model is None:
        st.error("couldn't load the modle, please ensure that it's in the same directory")
        return
    
    st.sidebar.title("input options :D")
    app_mode = st.sidebar.selectbox("choose input mode", ["Upload Image", "Webcam", "About"])
    
    if app_mode == "About":
        st.markdown("""
        ## about this website
        
        this website uses transfer learning with MobileNetV2 in order to detect emotions on people's faces. it uses haarCascade classifer in order to detect the faces themselves.
        
        ### the detected emotions are:
        - angry
        - happy
        - neutral
        - sad
        - surprised
        
        ### how to use:
        1. choose an upload image to analyze a photo
        2. choose the "webcam" option to use your camera in real-time
        3. this website will detect faces and predict emotions with confidence scores
        
        ### Model Details:
        - Base Model: MobileNetV2
        - Input Size: 224x224 pixels
        - Face Detection: Haar Cascade Classifier
        """)
    
    elif app_mode == "Upload Image":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader("choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("original Image")
                st.image(image_rgb, use_container_width=True)
            
            emotion, confidence, result_image = detect_emotion(image_rgb.copy(), model, face_cascade)
            
            with col2:
                st.subheader("detection result")
                if emotion is not None:
                    st.image(result_image, use_container_width=True)
                    
                    colors_hex = {
                        "Angry": "#F3B3B3",
                        "Happy": "#BCEEBC",
                        "Neutral": "#F7F797",
                        "Sad": "#AAAAEC",
                        "Surprised": "#EDCB8B"
                    }
                    
                    st.markdown(f"""
                        <div class="emotion-box" style="background-color: {colors_hex[emotion]}; color: white;">
                            {emotion}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("confidence", f"{confidence:.2f}%")
                    
                    if emotion in hampter_images:
                        st.subheader(f"{emotion} Hampter")
                        st.image(hampter_images[emotion], use_container_width=True)
                else:
                    st.warning("no face detected in the image, please choose a clear image of a front facing face in order for the face to be detected")
    
    elif app_mode == "Webcam":
        st.sidebar.markdown("---")
        st.sidebar.info("click start in order to begin webcam detection")
        
        run = st.sidebar.checkbox("start webcam")
        FRAME_WINDOW = st.image([])
        
        if run:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)
            
            if not cap.isOpened():
                st.error("can't open webcam, please check ur webcam/camera connection")
                return
            
            prediction_buffer = deque(maxlen=5)
            
            col1, col2 = st.columns([2, 1])
            emotion_placeholder = col2.empty()
            confidence_placeholder = col2.empty()
            hampter_placeholder = col2.empty()
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("failed to grab frame")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                current_emotion = "Neutral"
                current_confidence = 0.0
                
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame_rgb[y:y+h, x:x+w]
                    
                    facess = face_cascade.detectMultiScale(roi_gray)
                    
                    if len(facess) > 0:
                        for (ex, ey, ew, eh) in facess:
                            face_roi = roi_color[ey:ey+eh, ex:ex+ew]
                        
                        final_image = cv2.resize(face_roi, (224, 224))
                        final_image = np.expand_dims(final_image, axis=0)
                        final_image = preprocess_input(final_image)
                        
                        predictions = model.predict(final_image, verbose=0)
                        prediction_buffer.append(predictions[0])
                        
                        avg_predictions = np.mean(prediction_buffer, axis=0)
                        emotion_index = np.argmax(avg_predictions)
                        
                        emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]
                        current_emotion = emotions[emotion_index]
                        current_confidence = avg_predictions[emotion_index] * 100
                        
                        colors = {
                            "Angry": (255, 0, 0),
                            "Happy": (0, 255, 0),
                            "Neutral": (255, 255, 0),
                            "Sad": (0, 0, 255),
                            "Surprised": (255, 165, 0)
                        }
                        color = colors[current_emotion]
                        
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(frame_rgb, f"{current_emotion} ({current_confidence:.1f}%)", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                with col1:
                    FRAME_WINDOW.image(frame_rgb)
                
                with col2:
                    colors_hex = {
                        "Angry": "#F3B3B3",
                        "Happy": "#BCEEBC",
                        "Neutral": "#F7F797",
                        "Sad": "#AAAAEC",
                        "Surprised": "#EDCB8B"
                    }
                    
                    emotion_placeholder.markdown(f"""
                        <div class="emotion-box" style="background-color: {colors_hex[current_emotion]}; color: white;">
                            {current_emotion}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    confidence_placeholder.metric("Confidence", f"{current_confidence:.2f}%")
                    
                    if current_emotion in hampter_images:
                        hampter_placeholder.image(hampter_images[current_emotion], 
                                                caption=f"{current_emotion} Hampter",
                                                use_container_width=True)
            
            cap.release()
        else:
            st.info("click the start button on the sidebar in order to start the webcam")

if __name__ == "__main__":
    main()