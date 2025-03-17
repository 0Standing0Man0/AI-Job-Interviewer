# Author: Ambar Roy
# email: ambarroy11@gmail.com

from flask import Flask, render_template, Response, jsonify
import cv2
import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import threading
import wave
import pyaudio
import json
import time


from frames import extract_frames
from preprocess import transform_greyscale, ScharrEdgeDetection
from model import Model
from FramesDataset import FramesDataset
from whisper_model import transcribe_audio
from hugging_face_api import compute_score

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
processing = False
question_answer_counter = 0
used_questions = set()
all_questions = []
posture_score = 0.0
avg_comms_score = 0.0
indivisual_comms_score = []
display = False

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
    global recording, processing
    recording = False  # Reset state when page loads
    processing = False
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

    if not os.path.exists('Interview_Videos'):
        os.makedirs('Interview_Videos')
    if not os.path.exists('Interview_Audios'):
        os.makedirs('Interview_Audios')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('Interview_Videos/Interview_Video.mp4', fourcc, 20.0, (640, 480))   

    return jsonify({"message": "Recording Started", "recording": recording})

@app.route('/stop_recording')
def stop_recording():
    """
    Stops video and audio recording and transitions to processing.
    """
    global recording,audio_recording, used_questions, question_answer_counter, video_writer, processing
    recording = False  # Stop video recording
    audio_recording = False # Stop audio recording
    used_questions.clear()
    question_answer_counter=0

    if video_writer is not None:
        video_writer.release()
        video_writer = None
    
    processing = True
    return jsonify({"message": "Processing Interview", "processing": processing})

@app.route('/get_first_question')
def get_first_question():
    """
    Reads the first question from questions.txt and returns it as JSON.
    """
    global all_questions
    question = "No questions available."  # Default message if file is empty or missing

    if os.path.exists("questions.txt"):
        with open("questions.txt", "r") as file:
            lines = file.readlines()
            if lines:
                question = lines[0].strip()  # Get the first line (first question)
    
    all_questions.append(question)

    return jsonify({"question": question})

@app.route('/get_random_question')
def get_random_question():
    """
    Reads a random question from questions.txt and returns it as JSON.
    """
    global used_questions, all_questions

    if os.path.exists("questions.txt"):
        with open("questions.txt", "r") as file:
            lines = file.readlines()
            if lines:
                question = lines[0].strip()  # Get the first line (first question)
        
        available_indices = [i for i in range(1, min(len(lines), 11)) if i not in used_questions]  # Avoid out-of-range

        if available_indices:  # Ensure there are questions left
            chosen_index = random.choice(available_indices)  # Pick a valid index
            question = lines[chosen_index].strip()  # Get question text
            used_questions.add(chosen_index)  # Mark this index as used
            all_questions.append(question)

        else:
            return jsonify({"question": "No more unique questions available."})
    

    return jsonify({"question": question})

@app.route('/toggle_audio_recording')
def toggle_audio_recording():
    """
    Toggles the state of audio recording.
    """
    global recording, audio_recording, question_answer_counter
    audio_recording = not audio_recording  # Toggle state
    print(f"audio: {audio_recording}")

    if audio_recording:
            # Start audio recording in a separate thread
            audio_thread = threading.Thread(target=record_audio, args=(f"Interview_Audios/Answer_{question_answer_counter}.wav",))
            audio_thread.start()
    else:
            question_answer_counter+=1

    return jsonify({"audio_recording": audio_recording, "question_answer_counter": question_answer_counter})

@app.route('/processing_interview')
def processing_interview():
    """
    Processes the interview: extracts frames, evaluates posture, and performs speech-to-text.
    """
    global processing, posture_score, avg_comms_score, indivisual_comms_score, display

    start_time = time.time()

    all_answers = transcribe_audio()  # Converting speech to text
    print(type(all_questions),type(all_answers))
    print(all_questions)
    print(all_answers)

    # Write questions and answers to the file
    if not os.path.exists('Interview_Script'):
        os.makedirs('Interview_Script')
    
    if not os.path.exists('Interview_Script/Interview_Script.json') or os.stat('Interview_Script/Interview_Script.json').st_size == 0:
        with open('Interview_Script/Interview_Script.json', "w") as f:
            json.dump({"interview": []}, f)

    with open('Interview_Script/Interview_Script.json', "w") as f:
        json.dump({"interview": [{"question": q, "answer": a} for q, a in zip(all_questions, all_answers)]}, f)

    extract_frames('Interview_Videos/Interview_Video.mp4', 'Interview_Frames', 1)  # video path, output directory, fps
    
    # Load dataset
    frames_dataset = FramesDataset("Interview_Frames", transform_greyscale, ScharrEdgeDetection())
    frames_loader = DataLoader(frames_dataset, batch_size=1, shuffle=False)

    avg_comms_score, indivisual_comms_score = compute_score('Interview_Script/Interview_Script.json')

    posture_score = get_posture_score(frames_loader)

    end_time = time.time()

    print(f"Total Interview Analysis Time: {end_time - start_time} seconds")


    print(f"Posture is {'good' if posture_score >= 0.7 else 'bad'}!\nPosture Rating:{posture_score*100:.3f}%")
    print(f"Communication Skills: {avg_comms_score*100:.3f}%")
    print(f"Indivisual Q&A Score: {indivisual_comms_score}")


    # # # DELETING VIDEOS AND FRAMES # # #
    remove_interview_stuff('Interview_Videos')
    print("Interview video deleted")
    remove_interview_stuff('Interview_Frames')
    print("Interview frames deleted")
    remove_interview_stuff('Interview_Audios')
    print("Interview audios deleted")
    remove_interview_stuff('Interview_Script')
    print("Interview script deleted")
    
    
    processing = False
    display = True

    return jsonify({"message": "Processing Completed", "processing": processing, "display": display})

@app.route('/display_score')
def display_score():
    """
    Return the interview analysis scores, including posture score, average communication score, and individual communication scores.
    """
    global display, posture_score, avg_comms_score, indivisual_comms_score

    results = {
        "posture_score": posture_score,
        "average_communication_score": avg_comms_score,
        "individual_communication_scores": indivisual_comms_score
    }

    display = False

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
