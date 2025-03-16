import whisper
import os

def speech_to_text():
    # Ensure the directory exists
    if not os.path.isdir("Interview_Script"):
        os.makedirs("Interview_Script")

    # Load the Whisper model
    model = whisper.load_model("base")  # Change to "small", "medium", or "large" if needed

    # Check if the audio file exists before proceeding
    audio_path = "Interview_Audios/Interview.wav"
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
    else:
        # Transcribe the audio file
        result = model.transcribe(audio_path)

        # Save the transcription to a text file
        with open("Interview_Script/Interview.txt", "w", encoding="utf-8") as f:
            f.write(result["text"])

        # Print the transcribed text
        print(result["text"])
