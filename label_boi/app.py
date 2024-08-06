from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
FRAMES_FOLDER = 'frames'

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, FRAMES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get video duration
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        return jsonify({"filename": filename, "duration": duration})

@app.route('/fragment_video', methods=['POST'])
def fragment_video():
    data = request.json
    filename = data['filename']
    interval = int(data['interval'])
    frame_size = int(data['frame_size']) - 3
    start_time = data['start_time']
    end_time = data['end_time']

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(float(start_time) * fps)
    end_frame = int(float(end_time) * fps)

    if start_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT) or end_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return jsonify({"error": "The specified time range exceeds the video length."}), 400

    frame_count = 0
    saved_frame_count = 0
    frame_files = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
        if frame_count >= start_frame and frame_count % interval == 0:
            frame_filename = f'frame_{frame_count}.png'
            frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_files.append(frame_filename)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
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

if __name__ == '__main__':
    app.run(debug=True)