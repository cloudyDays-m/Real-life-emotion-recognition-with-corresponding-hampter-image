# Emotion detector with the corresponding hampter

This project shows real-time facial emotion recognition using a webcam and a MobileNetV2 based classifer. This program detects faces from the live video feed, and classifies the detected emotion, and then displays a corresponding hampter (meme hampster) image on the right side of the window. 

The face detection component was made using OpenCV's HaarCascade Classifier haarcascade_frontalface_default.xml. Each frame is converted to grayscale, and when the face is located, the ROI (Region of Interest) is cropped and resized to 224x224 pixels, and finally passed through the MobileNetV2 model for predicition. The model was built using Transfer Learning from the pre-trained MobileNetV2 architecture. The images were taken from the FER-2013 data set, and contains a customized 5 emotions in contrast to the originally organized 7. 

There is also an option to save screenshots of the window, which when pressing "s" creates (if not previously created) directory of screenshots saved in the directory where the repository is saved. This directory then has all of the screenshots taken from that window. 

### Keyboard controls: 

```p```: Pause or resume the webcam 

```s```: save a screenshot 

```q```: quit the program 

#### Emotion classes: 

Angry, Happy, Neutral, Sad, Suprised 

### Requirements: 

Python 3.8 or higher
TensorFlow / Keras
OpenCV
NumPy
Matplotlib (optional, for visualization)

# How to run? 

### Install dependencies: 

Make sure you have Python 3.8+ and install the required libraries 

### Clone this repository

Since the main file is in .ipynb, either download jupyter notebook through Anaconda or the jupyter extension on VScode. 

#### This is not nessesary but recomended to create and activiate a virtual environment: 
```
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
``` 



Extenstions → search for “Jupyter” and “Python” by Microsoft → install both

or

Install jupyter notebook through Anaconda (Anaconda is an open source data science and AI distribution plateform for Python and R) 

### Install the required dependencies: 

pip install -r requirements.txt

### Run the program: 
execute the main python script.

#### Controls:

Press P to pause/resume the live feed.

Press S to save a screenshot (stored in screenshots/ with a timestamp).

Press Q to quit the application.

It is gonna take a few mins, just let it run for a little bit :D 

#### Acknowledgments

This project’s model training and base architecture were developed with the help of the tutorial:
Realtime Face Emotion Recognition | Tensorflow | Transfer Learning | Python | Train your own Images (YouTube) (*https://www.youtube.com/watch?v=avv9GQ3b6Qg&t=120s*)

The tutorial by DeepLearning_by_PhDScholar on FER-2013 inspired the training process and provided foundational guidance for implementing transfer learning using MobileNetV2.
