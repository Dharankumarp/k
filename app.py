import os
import cv2
from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO("yolov5s.pt")

global video_path, max_person_count
video_path = None
max_person_count = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global video_path, max_person_count
    if 'video' not in request.files:
        return "No file part"

    file = request.files['video']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    return render_template('stream.html', filename=filename)

def generate_frames():
    global max_person_count
    cap = cv2.VideoCapture(video_path)
    max_person_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        person_count = sum(1 for cls in results[0].boxes.cls if int(cls) == 0)
        max_person_count = max(max_person_count, person_count)

        for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f'Persons: {person_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    return render_template('result.html', count=max_person_count)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
