from flask import Flask, render_template, Response, jsonify
import cv2
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import threading
import wave
import pyaudio

from frames import extract_frames
from preprocess import transform_greyscale, ScharrEdgeDetection
from model import Model
from FramesDataset import FramesDataset
from whisper_model import speech_to_text

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

# Function to record audio
def record_audio(audio_filename):
    """
    Records audio in a separate thread and saves it as a .wav file.
    """
    global audio_recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
    frames = []
    while audio_recording:
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    with wave.open(audio_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))

def remove_interview_stuff(folder):
    """
    Deletes all files in a given folder.
    """
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        # Ensure it's a file before deleting
        if os.path.isfile(file_path):
            os.remove(file_path)

app = Flask(__name__)
video_capture = None
audio_recording = False
recording = False
video_writer = None

def generate_frames():
    """
    Captures video frames from the webcam and streams them to the web page.
    """
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
    """
    Renders the home page.
    """
    global recording
    recording = False  # Reset state when page loads
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Streams video feed to the frontend.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    """
    Starts video and audio recording.
    """
    global recording, video_writer, audio_recording
    recording = True  # Enable video recording
    audio_recording = True  # Enable audio recording

    if not os.path.exists('Interviews'):
        os.makedirs('Interviews')
    if not os.path.exists('Interview_Audios'):
        os.makedirs('Interview_Audios')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('Interviews/Interview.mp4', fourcc, 20.0, (640, 480))

    # Start audio recording in a separate thread
    audio_thread = threading.Thread(target=record_audio, args=('Interview_Audios/Interview.wav',))
    audio_thread.start()

    return jsonify({"message": "Recording Started", "recording": recording})

@app.route('/stop_recording')
def stop_recording():
    """
    Stops video and audio recording, processes video frames, and evaluates posture.
    """
    global recording, video_writer, audio_recording
    recording = False  # Stop video recording
    audio_recording = False  # Stop audio recording

    if video_writer is not None:
        video_writer.release()
        video_writer = None
    extract_frames('Interviews/Interview.mp4', 'Interview_Frames', 1)  # video path, output directory, fps
    

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
