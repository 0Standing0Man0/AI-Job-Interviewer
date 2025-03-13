from flask import Flask, render_template, Response, jsonify
import cv2
import os

app = Flask(__name__)
video_capture = None
recording = False
video_writer = None

def generate_frames():
    global video_capture, recording, video_writer
    video_capture = cv2.VideoCapture(0)
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            if recording and video_writer is not None:
                video_writer.write(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global recording
    recording = False  # Reset state when page loads
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    global recording, video_writer
    recording = True  # Update state
    
    if not os.path.exists('Interviews'):
        os.makedirs('Interviews')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('Interviews/interview.mp4', fourcc, 20.0, (640, 480))
    
    return jsonify({"message": "Recording Started", "recording": recording})

@app.route('/stop_recording')
def stop_recording():
    global recording, video_writer
    recording = False  # Update state
    
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    
    return jsonify({"message": "Recording Stopped", "recording": recording})

if __name__ == '__main__':
    app.run(debug=True)
