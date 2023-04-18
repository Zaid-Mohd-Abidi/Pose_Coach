# import the necessary libraries
from flask import Flask, request, jsonify,render_template
import cv2
import numpy as np
import operator
import pygame
from keras.models import model_from_json

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model.h5")

# create a Flask app
app = Flask(__name__)

def model_pred(frame):
    result = loaded_model.predict(np.reshape(frame, [-1, 200, 200, 3]))
    predict =   { 'downdog':    result[0][0],
                  'goddess':    result[0][1],    
                  'plank':    result[0][2],
                  'tree':    result[0][3],
                  'warrior2':    result[0][4],
                  }
    
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    return predict
CLIP_X1 = 160
CLIP_Y1 = 140
CLIP_X2 = 400
CLIP_Y2 = 360

# create a route to handle incoming video feed
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test.html')

def process_video():
    # read the video file
    try:

        cap = cv2.VideoCapture(0)

        # initialize variables to count correct and incorrect frames
        correct_count = 0
        incorrect_count = 0

        # create a loop to read the frames
        while True:
            # read the frame
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1)

            # extract the region of interest (ROI) and preprocess it
            roi = frame[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2]
            roi = cv2.resize(roi, (200, 200)) 
            _, roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

            # apply the deep learning model to the ROI
            prediction = model_pred(roi)

            # increment the correct or incorrect count based on the prediction
            if prediction[0][1] > 0.5:
                correct_count += 1
            else:
                incorrect_count += 1

            # check if we have processed all frames
            if not cap.isOpened():
                break

        # calculate the percentage of correct frames
        total_frames = correct_count + incorrect_count
        percent_correct = (correct_count / total_frames) * 100

        # determine if the pose was held correctly for the majority of the video
        if percent_correct >= 50:
            output1 = 'correct'
        else:
            output1 = 'not correct'
    except Exception as e:
        print(e)

    # release the video capture and return the output
    cap.release()
    return render_template('test.html', output=output1)
    
        
if __name__=='__main__':
    app.run(debug=True)

