import whisper
import os

def transcribe_audio():
    # Ensure the directory exists
    if not os.path.isdir("Interview_Script"):
        os.makedirs("Interview_Script")

    # Load the Whisper model
    model = whisper.load_model("base")  # Change to "small", "medium", or "large" if needed

    # Check if the audio file exists before proceeding
    audio_path = "Interview_Audios/Interview.wav"
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        result = 0
    else:
        # Transcribe the audio file
        result = model.transcribe(audio_path)

        # Print the transcribed text
        print(result["text"])

    return result["text"]
