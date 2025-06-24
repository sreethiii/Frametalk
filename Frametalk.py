import os
import cv2
import numpy as np
import pyttsx3
from flask import Flask, request, render_template, send_from_directory, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded and processed images

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    # Default values when the page first loads
    return render_template('test1.html', detected_objects=[], output_image=None)

@app.route('/', methods=['POST'])
def upload_image():
    if 'image_file' not in request.files:
        return "No file part"

    file = request.files['image_file']
    if file.filename == '':
        return "No selected file"

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Load the image for object detection
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detection results
    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels, and collect unique detected object names
    detected_objects = set()  # Initialize detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.add(label)  # Add unique object names
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed image
    output_image_name = 'output_' + file.filename
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_name)
    cv2.imwrite(output_image_path, img)

    # Convert detected objects set to a list
    detected_objects = list(detected_objects)

    # Speak out loud about the detected objects
    if detected_objects:
        message = "I detected the following objects: " + ", ".join(detected_objects)
    else:
        message = "No objects were detected in the image."

    speak_text(message)  # Speak the message and wait for it to complete

    # Pass detected objects and processed image to the template
    return render_template(
        'test1.html',
        detected_objects=detected_objects,
        output_image=url_for('uploaded_file', filename=output_image_name)
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve the processed image
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def speak_text(text):
    try:
        engine = pyttsx3.init()  # Initialize a new engine instance for each request
        engine.say(text)
        engine.runAndWait()
        engine.stop()  # Ensure the engine stops after speaking
    except RuntimeError:
        print("Speech engine encountered an error.")

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
