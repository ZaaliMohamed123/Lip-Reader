# LipReader

## Description
LipReader is a deep learning application developed to perform lip reading from video input. The application utilizes a 3D convolutional neural network combined with LSTM layers to predict text from lip movements in videos. It is built using Streamlit for the user interface and TensorFlow for the model implementation.

## Installation
To run this project, you need to have Python installed along with the required packages. You can install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app/streamlitapp.py
   ```
2. Open the application in your web browser.
3. Select a video from the provided options to see the lip reading predictions.

## Project Structure
```
Lip reading project/
│
├── app/
│   ├── imgs/                # Contains images used in the application
│   ├── vids/                # Contains video files and generated outputs
│   ├── modelutil.py         # Contains model architecture and loading functions
│   ├── streamlitapp.py      # Main Streamlit application file
│   └── utils.py             # Utility functions for video processing and character mapping
│
├── data/                    # Directory containing video data for lip reading
│   └── alignments/          # Alignment data for videos
│
├── requirements.txt         # List of required Python packages
└── README.md                # Project documentation
```

## Acknowledgments
This application is developed by ZAALI Mohamed and SEKAL Douaâ. Special thanks to the contributors and libraries that made this project possible.
