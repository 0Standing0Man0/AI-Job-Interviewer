import os
import subprocess

def extract_audio(video_path, audio_path):
    if not os.path.exists("Interview_Audios"):
        os.makedirs("Interview_Audios")
    
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path}"
    subprocess.run(command, shell=True)

extract_audio('Interviews/Interview.mp4', 'Interview_Audios/Interview.mp3')