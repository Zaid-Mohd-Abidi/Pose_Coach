# import the necessary libraries
from flask import Flask, request, jsonify,render_template
import cv2
import numpy as np
import operator
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
    # get the video feed from the client
    #video_file = request.files['video']

    # read the video file
    cap = cv2.VideoCapture(0)

    # create a loop to read the frames
    while True:
        #ret, frame = video.read()
        _, FrameImage = cap.read()
        FrameImage = cv2.flip(FrameImage, 1)
        #cv2.imshow("", FrameImage)
        cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2, CLIP_Y2), (0,255,0) ,1)

        ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2]
        ROI = cv2.resize(ROI, (200, 200)) 
        #ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        #ROI= cv2.add(ROI,np.array([40.0]))
        _, output = cv2.threshold(ROI, 200, 255, cv2.THRESH_BINARY) # adjust brightness
        

        # apply the deep learning model to the frame
        predict=model_pred(ROI)
        
        #if predict[0][1]>1.0:
        return render_template('test.html',output=predict[0][0])
        #else:
            #return render_template('test.html',output='not correct')
    #return render_template('test.html')
if __name__=='__main__':
    app.run(debug=True)

