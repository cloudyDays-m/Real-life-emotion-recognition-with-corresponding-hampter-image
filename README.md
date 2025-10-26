# Emotion detector with the corresponding hampter

This project shows real-time facial emotion recognition using a webcam and a MobileNetV2 based classifer. This program detects faces from the live video feed, and classifies the detected emotion, and then displays a corresponding hampter (meme hampster) image on the right side of the window. 

The face detection component was made using OpenCV's HaarCascade Classifier haarcascade_frontalface_default.xml. Each frame is converted to grayscale, and when the face is located, the ROI (Region of Interest) is cropped and resized to 224x224 pixels, and finally passed through the MobileNetV2 model for predicition. The model was built using Transfer Learning from the pre-trained MobileNetV2 architecture. The images were taken from the FER-2013 data set, and contains a customized 5 emotions in contrast to the originally organized 7. 

#### Emotion classes: 

Angry, Happy, Neutral, Sad, Suprised 

### Requirements: 

Python 3.8 or higher
TensorFlow / Keras
OpenCV
NumPy
Matplotlib (optional, for visualization)

# How to run? 

Deployable link made with streamlit: https://cloudydays-m-real-life-emotion-recognition-with-corr-app-rtaw0i.streamlit.app/ 

#### Acknowledgments

This project’s model training and base architecture were developed with the help of the tutorial:
Realtime Face Emotion Recognition | Tensorflow | Transfer Learning | Python | Train your own Images (YouTube) (*https://www.youtube.com/watch?v=avv9GQ3b6Qg&t=120s*)

The tutorial by DeepLearning_by_PhDScholar on FER-2013 inspired the training process and provided foundational guidance for implementing transfer learning using MobileNetV2.

<img width="1911" height="901" alt="image" src="https://github.com/user-attachments/assets/575811e7-1fb9-47dd-bee3-b3f7c8d70460" />

<img width="1920" height="892" alt="image" src="https://github.com/user-attachments/assets/f803b524-4eaa-419b-94da-71b35171c740" />



[![Athena Award Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Faward.athena.hackclub.com%2Fapi%2Fbadge)](https://award.athena.hackclub.com?utm_source=readme)


