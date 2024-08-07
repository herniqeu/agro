from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
import cv2
import numpy as np
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'files/uploads'
PROCESSED_FOLDER = 'files/processed'
FRAMES_FOLDER = 'files/frames'
SEGMENTED_FOLDER = 'files/segmented'

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, FRAMES_FOLDER, SEGMENTED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processed/<path:filename>')
def processed_files(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/segmented/<path:filename>')
def segmented_files(filename):
    return send_from_directory(SEGMENTED_FOLDER, filename)

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
    frames_per_second = int(data['frames_per_second'])
    frame_size = int(data['frame_size']) - 3
    start_time = data['start_time']
    end_time = data['end_time']

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(filepath)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(float(start_time) * video_fps)
    end_frame = int(float(end_time) * video_fps)

    if start_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT) or end_frame > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        return jsonify({"error": "The specified time range exceeds the video length."}), 400

    frame_interval = int(video_fps / frames_per_second)
    frame_count = 0
    saved_frame_count = 0
    frame_files = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
        if frame_count >= start_frame and frame_count % frame_interval == 0:
            frame_time = frame_count / video_fps
            frame_filename = f'{os.path.splitext(filename)[0]}_{frame_time:.2f}.png'
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
    original_width = int(data['originalWidth'])
    original_height = int(data['originalHeight'])

    filepath = os.path.join(FRAMES_FOLDER, filename)
    img = Image.open(filepath)
    
    # Ensure the crop area is within the image bounds
    x = max(0, min(x, original_width - width))
    y = max(0, min(y, original_height - height))
    
    cropped_img = img.crop((x, y, x + width, y + height))
    
    frame_size = int(request.form.get('frame_size', 60))
    cropped_img = cropped_img.resize((frame_size, frame_size), Image.LANCZOS)
    
    processed_filename = f'{os.path.splitext(filename)[0]}_processed.png'
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    cropped_img.save(processed_path)
    
    return jsonify({"message": "Concluído!"})

@app.route('/get_processed_images')
def get_processed_images():
    images = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify({"images": images})

@app.route('/apply_segmentation', methods=['POST'])
def apply_segmentation():
    data = request.json
    image_data = data['image'].split(',')[1]
    filename = data['filename']

    # Decode base64 image
    img_data = base64.b64decode(image_data)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"message": "Failed to decode image."}), 400

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for red color and create a mask
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Create an output image with red traces in black and the rest in white
    output = cv2.bitwise_not(mask)

    # Save the segmented image
    segmented_filename = f'{os.path.splitext(filename)[0]}_segmented.png'
    segmented_path = os.path.join(SEGMENTED_FOLDER, segmented_filename)
    cv2.imwrite(segmented_path, output)
    return jsonify({"message": "Segmentation applied successfully!"})

if __name__ == '__main__':
    app.run(debug=True)