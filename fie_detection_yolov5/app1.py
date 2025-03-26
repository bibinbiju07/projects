from flask import Flask, render_template, Response, request, jsonify
import torch
import cv2
import numpy as np
from PIL import Image
import pygame
import os
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# Folders for uploaded and processed files
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.eval()

# Initialize pygame for alarm sound
pygame.mixer.init()
alarm_sound = "fire_alarm.mp3"  # Replace with actual alarm sound file path
alarm_active = False
fire_detected_once = False
video_streaming = False

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold for detection
ALERT_LABEL = ['fire'] 
alarm_playing = False  # Flag to track if the alarm is playing
alarm_thread = None  # Thread for playing alarm sound

# Email Configuration
EMAIL_ADDRESS = "fire.detector.danger@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "pjyqehmgmdbviose"  # Replace with your email password
TO_EMAIL = "bibinbiju0247@gmail.com"  # Replace with recipient email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def send_email():
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = "🔥 Fire Detected! 🔥"
        body = "Alert! Fire has been detected. Please take immediate action."
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        server.quit()
        print("Email notification sent!")
    except Exception as e:
        print("Failed to send email:", e)

def play_alarm():
    global alarm_active, fire_detected_once
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play(-1)
        alarm_active = True
        fire_detected_once = True
        threading.Thread(target=send_email).start()  # Run send_email in a separate thread

def stop_alarm():
    global alarm_active, fire_detected_once
    pygame.mixer.music.stop()
    alarm_active = False
    fire_detected_once = False

def generate_frames():
    global alarm_active, fire_detected_once
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(image)
        df = results.pandas().xyxy[0]
        fire_detected = False

        for _, row in df.iterrows():
            confidence = row['confidence']
            class_name = row['name'].lower()
            if confidence > 0.5 and class_name == 'fire':
                fire_detected = True
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Play alarm in a separate thread to avoid blocking video stream
        if fire_detected and not fire_detected_once:
            fire_detected_once = True  # Ensure alarm plays only once
            threading.Thread(target=play_alarm, daemon=True).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    
def detect_objects(image_path, output_path):
    """Detects objects in an image and plays the alarm if needed."""
    global alarm_playing, alarm_thread

    image = Image.open(image_path)
    results = model(image)
    df = results.pandas().xyxy[0]

    img = cv2.imread(image_path)
    alert_triggered = False

    for _, row in df.iterrows():
        confidence = row['confidence']
        class_name = row['name']
        if confidence >= CONFIDENCE_THRESHOLD and class_name in ALERT_LABEL:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            alert_triggered = True

    if alert_triggered and not alarm_playing:
        alarm_thread = threading.Thread(target=play_alarm, daemon=True)
        alarm_thread.start()

    cv2.imwrite(output_path, img)
    return output_path


def detect_video_stream(video_path):
    """Processes a video stream and plays the alarm only if the detected animal is in the alert list."""
    global alarm_playing, alarm_thread

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(image)
        df = results.pandas().xyxy[0]

        alert_triggered = False

        for _, row in df.iterrows():
            confidence = row['confidence']
            class_name = row['name']
            if confidence >= CONFIDENCE_THRESHOLD and class_name in ALERT_LABEL:
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                alert_triggered = True

        if alert_triggered and not alarm_playing:
            alarm_thread = threading.Thread(target=play_alarm, daemon=True)
            alarm_thread.start()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    stop_alarm()  # Stop alarm when the video ends

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live-stream')
def live_stream():
    return render_template('live_stream.html')

@app.route('/file-upload')
def file_upload():
    return render_template('file_upload.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and processes images or videos."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({"video_url": file_path})
    else:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.jpg')
        detect_objects(file_path, output_path)
        return jsonify({"image_url": output_path})


@app.route('/detect_video')
def detect_video():
    """Streams video with real-time detection."""
    video_path = request.args.get('video_path')
    return Response(detect_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    stop_alarm()
    return "Alarm stopped"

if __name__ == '__main__':
    app.run(debug=True)
