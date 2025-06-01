# Models
# hf.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF:Q4_0 = Llama 3
# hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:latest = SmolLM2
# hf.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_0     = TinyLlama
import ollama

MODEL = 'hf.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_0'  # Tinylama

import cv2
import numpy as np
import sounddevice as sd
import threading
import time
import queue
import os
from collections import deque
from scipy.io.wavfile import write
from scipy.signal import resample
import speech_recognition as sr
import pyttsx3
import subprocess
import imageio_ffmpeg

# === CONFIG ===
FPS = 20
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BUFFER_SECONDS = 60
AUDIO_SAMPLE_RATE = 44100

# === TIMING & BUFFERS ===
start_time = time.perf_counter()
video_buffer = deque(maxlen=FPS * BUFFER_SECONDS)
audio_buffer = deque()
audio_queue = queue.Queue()


# === AUDIO STREAM ===
def audio_callback(indata, frames, time_info, status):
    timestamp = time.perf_counter() - start_time
    audio_queue.put((timestamp, indata.copy()))


def audio_recording_loop():
    with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=1024):
        while True:
            ts, data = audio_queue.get()
            audio_buffer.append((ts, data))


threading.Thread(target=audio_recording_loop, daemon=True).start()

# === VIDEO STREAM ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


def video_capture_loop():
    interval = 1 / FPS
    while True:
        start = time.perf_counter()
        ret, frame = cap.read()
        if ret:
            timestamp = time.perf_counter() - start_time
            video_buffer.append((timestamp, frame))
        elapsed = time.perf_counter() - start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


threading.Thread(target=video_capture_loop, daemon=True).start()


# === CLIP SAVING ===
def save_clip():
    print('Josh> Saving Clip')
    pyttsx3.speak('Saving clip!')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_video = f"clip_{timestamp}.avi"
    raw_audio = f"clip_{timestamp}.wav"
    final_output = f"clip_{timestamp}.mp4"

    print("[*] Saving synchronized clip...")

    # Extract video data
    video_data = list(video_buffer)
    if not video_data:
        print("[!] No video captured.")
        return

    start_ts = video_data[0][0]
    end_ts = video_data[-1][0]
    video_duration = end_ts - start_ts
    print(f"[*] Video duration: {video_duration:.2f}s")

    # === Save Video ===
    out = cv2.VideoWriter(raw_video, cv2.VideoWriter_fourcc(*'XVID'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    for _, frame in video_data:
        out.write(frame)
    out.release()
    print("[✓] Video saved.")

    # === Extract & Resample Audio ===
    AUDIO_OFFSET = 0.4  # seconds, adjust as needed

    relevant_audio = [
        data for ts, data in audio_buffer
        if start_ts + AUDIO_OFFSET <= ts <= end_ts + AUDIO_OFFSET
    ]
    if not relevant_audio:
        print("[!] No matching audio.")
        return

    audio_np = np.concatenate(relevant_audio).flatten()
    target_samples = int(video_duration * AUDIO_SAMPLE_RATE)
    audio_resampled = resample(audio_np, target_samples)
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    write(raw_audio, AUDIO_SAMPLE_RATE, audio_int16)
    print("[✓] Audio saved.")

    # === Merge with FFmpeg ===
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path, "-y",
        "-i", raw_video,
        "-i", raw_audio,
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-shortest",
        final_output
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] Final clip saved: {final_output}")
        speak_text = 'Clip saved as', final_output
        pyttsx3.speak(speak_text)
    except subprocess.CalledProcessError as e:
        print(f"[!] FFmpeg error: {e}")
    finally:
        try:
            os.remove(raw_video)
            os.remove(raw_audio)
        except Exception as e:
            print(f"[!] Cleanup error: {e}")


# === VOICE COMMANDS ===
recognizer = sr.Recognizer()


def voice_command_loop():
    pyttsx3.speak('Activated!')
    print("Activated")

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source)

            command = recognizer.recognize_google(audio).lower()
            if "josh clip" in command:
                threading.Thread(target=save_clip, daemon=True).start()
            elif "josh" in command:
                print("You said:", command)
                stream = ollama.chat(
                    model=MODEL,
                    messages=[{'role': 'user', 'content': command}],
                    stream=True
                )
                answer = ''
                for chunk in stream:
                    answer += chunk['message']['content']
                print(answer)
                pyttsx3.speak(answer)

        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[ERROR] Speech recognition: {e}")


threading.Thread(target=voice_command_loop, daemon=True).start()

# === MAIN LOOP ===
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()
