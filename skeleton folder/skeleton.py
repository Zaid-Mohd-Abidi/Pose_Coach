from flask import Flask,render_template,Response
import cv2
import mediapipe as mp
import numpy as np
import os
import json
from keras.models import load_model
from keras.models import model_from_json
import operator

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

IMG_FOLDER = os.path.join('static', 'image')

json_file = open('C:/Users/shiv taneja/capstone project/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/shiv taneja/capstone project/model.h5")
def model_pred(frame):
    result = loaded_model.predict(np.reshape(frame, [-1, 200, 200, 3]))
    predict =   { 'downdog':    result[0][0],
                        'goddess':    result[0][1],    
                        'plank':    result[0][2],
                        'tree':    result[0][3],
                        'warrior2':    result[0][4],
                        }
            
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    return predict[0][0]    

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
image_filenames = [filename for filename in os.listdir(IMG_FOLDER) if filename.endswith('.jpg')]

camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process the RGB frame to get the result
            results = pose.process(RGB)

            print(results.pose_landmarks)
            # draw detected skeleton on the frame
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
       

        yield(b'--frame\r\n'
    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

        

'''@app.route('/video')
def index():
    return render_template('sample3.html')'''
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/index.html')
def Display_IMG():
    return render_template("index.html", image_filenames=image_filenames)
@app.route('/yoga.html')
def yoga():
    return render_template('yoga.html')
@app.route('/')
def home():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)