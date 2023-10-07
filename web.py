import streamlit as st
from io import BytesIO
from PIL import Image
import keras
import numpy as np
import librosa

class livePredictions:
    def __init__(self, path, file):
        self.path = path
        self.file = file

    def load_model(self):
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
#         predictions = self.loaded_model.predict_classes(x)
        predict_x=self.loaded_model.predict(x) 
        predictions=np.argmax(predict_x,axis=1)
#         predictions = np.argmax(loaded_model.predict(x_test),axis=1)
        return self.convertclasstoemotion(predictions)

    @staticmethod
    def convertclasstoemotion(pred):
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label
    
def detect_emotion(audio_file):
        if audio_file is not None:
            pred = livePredictions(path='Audio_SER.h5',file=audio_file)
            pred.load_model()
            emotion=pred.makepredictions() 
            return emotion
st.title("Audio Emotion Detection App")
st.subheader("Upload an audio file, and I'll detect the emotion in it.")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])    

if audio_file:
    emotion = detect_emotion(audio_file)
    
    if emotion:
        if emotion == "happy":
            emotion = "Happy"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/happy.jpg") 
        elif emotion == "angry":
            emotion = "Angry"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/angry.png")  
        elif emotion == "calm":
            emotion = "Calm"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/calm.png")
        elif emotion == "sad":
            emotion = "Sad"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/sad.png")
        elif emotion == "fearful":
            emotion = "Fearful"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/     fearful.jpg")
        elif emotion == "disgust":
            emotion = "Disgust"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/disgust.jpg")
        elif emotion == "surprised":
            emotion = "Surprised"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/surprised.png")   
        else:
            emotion = "Neutral"
            st.subheader(f"Emotion detected: {emotion}")
            emotion_image = Image.open("assets/neutral1.png")  

        st.image(emotion_image, caption=f"Emotion: {emotion}", use_column_width=True)

