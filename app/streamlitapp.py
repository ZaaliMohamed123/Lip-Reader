#import all the dependencies
import streamlit as st 
import os 
import imageio
import tensorflow as tf
import numpy as np
from utils import num_to_char,load_video
from modelutil import load_model


#Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

#Setup a sidebar
with st.sidebar:
    st.image('imgs\\logo.jpeg')
    st.title('LipReader')
    st.info('This application is developed by ZAALI Mohamed and SEKAL Douaâ')
    
st.title('Lip Reading Application')
#Generating a list of options or videos (this block of code can be replaced for to work with a webcam or some future solutions)
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Select a video',options)

#split the web page to two columns
col1, col2 = st.columns(2)

if options:
    
    #this column will display the video
    with col1:
        st.info('The video')
        #file_path for the selected video
        file_path = os.path.join('..','data','s1',selected_video)
        #os.system will allow us to run a command line call
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 vids\\test_video.mp4 -y') # this command allows us to use a ffmpeg library that will convert the videos from .mpg to .mp4
        
        #Rendering inside of the app
        video = open('vids\\test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
        
    #this column will display what the video will go through from preprocessing to prediction result with the DL model    
    with col2:
        st.info('This is what the Deep learning model sees when making predictions')
        video = load_video(file_path)
        
        #convert the video into a gif
        video_np = video.numpy()  # Convert to NumPy array
        video_np = np.squeeze((video_np * 255).astype(np.uint8))
        imageio.mimsave('vids\\animation.gif', video_np, fps=10)

        # Render the GIF inside the app
        st.image('vids\\animation.gif', width=500)
        
        
        st.info('This is the raw output of the model')
        model = load_model()
        output = model.predict(tf.expand_dims(video,axis=0))
        st.text(tf.argmax(output,axis=1).numpy())
        
        st.info('This is the output of the model after getting processed using CTC decoder') 
        decoder = tf.keras.backend.ctc_decode(output, input_length=[75], greedy=True)[0][0].numpy()
        st.text(decoder)
        
        st.info('The decoded output is converted into readable text.')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

    
