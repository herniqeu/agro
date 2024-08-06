from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
FRAMES_FOLDER = 'frames'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)
if not os.path.exists(FRAMES_FOLDER):
    os.makedirs(FRAMES_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filename  # Retorna o nome do arquivo do vídeo

@app.route('/fragment_video', methods=['POST'])
def fragment_video():
    data = request.json
    filename = data['filename']
    interval = int(data['interval'])
    start_time = data['start_time']
    end_time = data['end_time']

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # Convert start and end times to seconds
    start_minutes, start_seconds = map(int, start_time.split(':'))
    end_minutes, end_seconds = map(int, end_time.split(':'))
    start_frame = (start_minutes * 60 + start_seconds) * fps
    end_frame = (end_minutes * 60 + end_seconds) * fps

    if start_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT) or end_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return jsonify({"error": "The specified time range exceeds the video length."})

    frame_count = 0
    saved_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
        if frame_count >= start_frame and frame_count % interval == 0:
            frame_path = os.path.join(FRAMES_FOLDER, f'frame_{frame_count}.png')
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    frame_files = sorted([f for f in os.listdir(FRAMES_FOLDER) if f.endswith('.png')])
    return jsonify({"frames": frame_files})

@app.route('/frames/<filename>')
def get_frame(filename):
    return send_from_directory(FRAMES_FOLDER, filename)

@app.route('/crop', methods=['POST'])
def crop_image():
    data = request.json
    filename = data['filename']
    x = int(data['x'])
    y = int(data['y'])
    width = int(data['width'])
    height = int(data['height'])

    filepath = os.path.join(FRAMES_FOLDER, filename)
    img = Image.open(filepath)
    cropped_img = img.crop((x, y, x + width, y + height))
    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    cropped_img.save(processed_path)
    return jsonify({"message": "Concluído!"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)