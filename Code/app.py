from flask import Flask, render_template, Response
import cv2
import torch
from PIL import Image
from model import Model
from preprocess import transform_greyscale, edge_detection

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
saved_model = Model(num_classes=2).to(device)
saved_model.load_state_dict(torch.load('posture_AlexNet.pth', map_location=device))

app = Flask(__name__)
app.template_folder = 'templates'

is_recording = False

def video_stream():
    cap = cv2.VideoCapture(0)
    out = None  # Initialize the video writer

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale tensor
        frame_tensor = transform_greyscale(Image.fromarray(frame))

        # Apply edge detection
        edge_frame = edge_detection(frame_tensor)

        # Encode original frame
        ret, buffer_original = cv2.imencode('.jpg', frame)
        original_frame_bytes = buffer_original.tobytes()

        # Yield original frame with appropriate header
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + original_frame_bytes + b'\r\n\r\n')

        if is_recording:  # Check if recording is active
            if out is None:
                # Start recording
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter('video.mp4', fourcc, 6.0, (frame.shape[1], frame.shape[0]))
            out.write(frame)

            # Get predictions
            predicted = []
            with torch.no_grad():
                outputs = saved_model(edge_frame)
                _, predicted = torch.max(outputs.data, 1)
                predicted.extend(predicted.cpu().numpy())
            
            # Prepare the text overlay
            if predicted[0] == 1:
                text = "Valid Posture"
            else:
                text = "Invalid Posture"

            yield text

        # ... (rest of the video streaming)

    if out is not None:
        out.release()
    cap.release()

@app.route('/')
def index():
    print(f"Rendering template: index.html")  # Add this line
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)