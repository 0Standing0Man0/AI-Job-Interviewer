import whisper
import os

def transcribe_audio():

    # Load the Whisper model
    model = whisper.load_model("base")  # Change to "small", "medium", or "large" if needed

    audio_dir = "Interview_Audios"
    transcriptions = []

    # Ensure the directory exists
    if not os.path.exists(audio_dir):
        print(f"Error: Directory '{audio_dir}' not found.")
        return []

    # Loop through all files in the directory
    for filename in sorted(os.listdir(audio_dir)):  # Sort to maintain order
        file_path = os.path.join(audio_dir, filename)

        # Process only audio files
        if os.path.isfile(file_path) and file_path.lower().endswith((".wav", ".mp3", ".m4a")):
            print(f"Transcribing: {filename}")
            result = model.transcribe(file_path, language="en")
            transcriptions.append(result["text"])  # Store transcribed text

    return transcriptions  # Return the list of transcriptions

'''
print(transcribe_audio())
'''