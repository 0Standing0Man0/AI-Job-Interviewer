from flask import Flask, render_template, Response, jsonify
import cv2
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import subprocess

from frames import extract_frames
from preprocess import transform_greyscale, ScharrEdgeDetection
from model import Model
from FramesDataset import FramesDataset

# Set device
device = torch.device('cpu')

def get_posture_score(dataloader):
    # Load trained model
    model = Model(num_classes=2).to(device)
    model.load_state_dict(torch.load("posture_VGGNet16.pth", map_location=torch.device("cpu")), strict=False)
    model.eval()

    # Perform predictions
    all_preds = []
    with torch.no_grad():
        for edge_img in dataloader:
            edge_img = edge_img.to(device)
            outputs = model(edge_img)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.item())

    return np.mean(all_preds)

def extract_audio(video_path, audio_path):
    if not os.path.exists("Interview_Audios"):
        os.makedirs("Interview_Audios")
    
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}"
    subprocess.run(command, shell=True)

def remove_interview_stuff(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        # Ensure it's a file before deleting
        if os.path.isfile(file_path):
            os.remove(file_path)

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
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
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
    video_writer = cv2.VideoWriter('Interviews/Interview.mp4', fourcc, 20.0, (640, 480))
    
    return jsonify({"message": "Recording Started", "recording": recording})

@app.route('/stop_recording')
def stop_recording():
    global recording, video_writer
    recording = False  # Update state
    
    if video_writer is not None:
        video_writer.release()
        video_writer = None

    extract_frames('Interviews/Interview.mp4','Interview_Frames',1) # video path, output directory, fps
    # extract_audio('Interviews/Interview.mp4', 'Interview_Audios/Interview.mp3')

    # Load dataset
    frames_dataset = FramesDataset("Interview_Frames", transform_greyscale, ScharrEdgeDetection())
    frames_loader = DataLoader(frames_dataset, batch_size=1, shuffle=False)

    Posture_Score = get_posture_score(frames_loader)
    print(f"Posture is {'good' if np.round(Posture_Score) == 1 else 'bad'}!")


    '''
    # # # DELETING VIDEOS AND FRAMES # # #
    remove_interview_stuff('Interviews')
    print("Interview video deleted")
    remove_interview_stuff('Interview_Frames')
    print("Interview frames deleted")
    '''
    
    return jsonify({"message": "Recording Stopped", "recording": recording})

if __name__ == '__main__':
    app.run(debug=True)
