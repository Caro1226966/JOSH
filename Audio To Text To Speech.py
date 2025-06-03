# Models
# hf.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF:Q4_0 = Llama 3
# hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:latest = SmolLM2
# hf.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_0     = TinyLlama

MODEL = 'hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:latest'  # #Smollm2

import ollama
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
import pyttsx3
import subprocess
import imageio_ffmpeg
from vosk import Model as VoskModel, KaldiRecognizer
import json

# === CONFIG ===
FPS = 20
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
BUFFER_SECONDS = 60
AUDIO_SAMPLE_RATE = 44100
MEMORY_FILE = "memory.txt"
ended = False


def terminate_josh():
    print("Good Bye!")
    pyttsx3.speak('Good Bye')
    save_memory(conversation_history)
    cap.release()
    cv2.destroyAllWindows()
    exit()

# === MEMORY ===
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return [{
            "role": "system",
            "content": "You are Josh, a helpful assistant. You like to keep responses medium size and are a bit curious. NEVER USE EMOJIS! Use opinions and feelings based on the conversation"
        }]

    history = []
    with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n\n')
        for entry in lines:
            if 'role:' in entry and 'content:' in entry:
                role_line = entry.strip().split('\n')[0]
                content_line = '\n'.join(entry.strip().split('\n')[1:])
                role = role_line.replace('role:', '').strip()
                content = content_line.replace('content:', '').strip()
                history.append({"role": role, "content": content})
    return history


def save_memory(history):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        for item in history:
            f.write(f"role: {item['role']}\n")
            f.write(f"content: {item['content']}\n\n")


conversation_history = load_memory()

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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_video = f"clip_{timestamp}.avi"
    raw_audio = f"clip_{timestamp}.wav"
    final_output = f"clip_{timestamp}.mp4"

    print("[*] Saving synchronized clip...")
    video_data = list(video_buffer)
    if not video_data:
        print("[!] No video captured.")
        return

    start_ts = video_data[0][0]
    end_ts = video_data[-1][0]
    video_duration = end_ts - start_ts
    print(f"[*] Video duration: {video_duration:.2f}s")

    out = cv2.VideoWriter(raw_video, cv2.VideoWriter_fourcc(*'XVID'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    for _, frame in video_data:
        out.write(frame)
    out.release()
    print("[✓] Video saved.")

    AUDIO_OFFSET = 0.4
    relevant_audio = [data for ts, data in audio_buffer if start_ts + AUDIO_OFFSET <= ts <= end_ts + AUDIO_OFFSET]
    if not relevant_audio:
        print("[!] No matching audio.")
        return

    audio_np = np.concatenate(relevant_audio).flatten()
    target_samples = int(video_duration * AUDIO_SAMPLE_RATE)
    audio_resampled = resample(audio_np, target_samples)
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    write(raw_audio, AUDIO_SAMPLE_RATE, audio_int16)
    print("[✓] Audio saved.")

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
        pyttsx3.speak(f'Clip saved as {final_output}')
    except subprocess.CalledProcessError as e:
        pyttsx3.speak('There was an error clipping! FFmpeg error')
        print(f"[!] FFmpeg error: {e}")
    finally:
        try:
            os.remove(raw_video)
            os.remove(raw_audio)
        except Exception as e:
            print(f"[!] Cleanup error: {e}")


# === VOICE COMMANDS with Vosk and Memory ===
vosk_model = VoskModel("models/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(vosk_model, AUDIO_SAMPLE_RATE)


def voice_command_loop_vosk():
    pyttsx3.speak('Activated!')
    print("Activated")

    def callback(indata, frames, time_info, status):
        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            command = result.get("text", "").lower()
            if not command:
                return

                # Voice command to shut josh down
            if command == 'josh terminate code two zero zero nine':
                print("Good Bye!")
                pyttsx3.speak('Good Bye')
                cap.release()
                cv2.destroyAllWindows()
                save_memory(conversation_history)
                terminate_josh()
                # Voice command to fully wipe josh's memory. BE CAREFUL!
            elif command == 'josh clear memory code two zero zero nine':
                if os.path.exists(MEMORY_FILE):
                    os.remove(MEMORY_FILE)
                    print("File deleted.")
                    pyttsx3.speak('Memory Cleared')
                    terminate_josh()
                else:
                    print("File does not exist.")
                    pyttsx3.speak('There was no memory to clear')

                # Voice command to clip the last 60 seconds
            elif "josh clip" in command:
                print('Clipping...')
                pyttsx3.speak('Clipping')
                threading.Thread(target=save_clip, daemon=True).start()

                # Voice trigger for the ai
            elif "josh" in command:
                print("You:", command)
                conversation_history.append({"role": "user", "content": command})

                stream = ollama.chat(
                    model=MODEL,
                    messages=conversation_history,
                    stream=True
                )

                answer = ''
                for chunk in stream:
                    answer += chunk['message']['content']

                conversation_history.append({"role": "assistant", "content": answer})
                save_memory(conversation_history)
                print(answer)
                pyttsx3.speak(answer)

    with sd.RawInputStream(samplerate=AUDIO_SAMPLE_RATE, blocksize=8000, dtype='int16', channels=1, callback=callback):
        while True:
            time.sleep(0.1)


threading.Thread(target=voice_command_loop_vosk, daemon=True).start()

# === MAIN LOOP ===
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass


