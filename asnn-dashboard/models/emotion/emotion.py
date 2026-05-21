import numpy as np
import os
import argparse
import sys
import cv2 as cv
import time
from asnn.api import asnn
from asnn.types import *

# Emotion classes (standard FER-2013 order)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Haar cascade for face detection
FACE_CASCADE = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

if FACE_CASCADE.empty():
    sys.exit("Error: Could not load haarcascade_frontalface_default.xml")

os.environ["QT_QPA_PLATFORM"] = "xcb"

def preprocess_face(face_gray):
    """Resize to 48x48, normalize to [0,1], add batch and channel dims"""
    face = cv.resize(face_gray, (48, 48))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)        # batch dimension
    face = np.expand_dims(face, axis=-1)       # channel dimension: (1,48,48,1)
    face = face.transpose(0, 3, 1, 2)          # NCHW format: (1,1,48,48)
    return [face]  # ASNN expects list of inputs

def postprocess_emotion(output_data):
    """Get emotion prediction from ASNN output"""
    # output_data is list with one tensor: shape (1, 7)
    preds = output_data[0].reshape(7)
    confidence = np.max(preds)
    emotion_idx = np.argmax(preds)
    return EMOTION_LABELS[emotion_idx], confidence

def draw_emotion(frame, x, y, w, h, emotion, confidence):
    """Draw bounding box and emotion label"""
    # Bounding box
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Label
    label = f"{emotion} ({confidence:.2f})"
    (label_width, label_height), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    # Background for label
    label_y = y - 10 if y - 10 > label_height else y + h + label_height + 10
    cv.rectangle(frame, (x, label_y - label_height - 10),
                 (x + label_width, label_y + baseline - 10),
                 (255, 255, 255), cv.FILLED)
    
    # Text
    cv.putText(frame, label, (x, label_y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time Facial Emotion Recognition using ASNN")
    parser.add_argument("--library", required=True, help="Path to ASNN C static library file")
    parser.add_argument("--model", required=True, help="Path to converted emotion .nbg model file")
    parser.add_argument("--video", default=None,
                        help="Path to input video file. Use '0' or omit for webcam.")
    parser.add_argument("--level", choices=['0', '1', '2'], default='0',
                        help="ASNN info level: 0/1/2")
    args = parser.parse_args()

    # Validate files
    if not os.path.exists(args.library):
        sys.exit(f'Library not found: {args.library}')
    if not os.path.exists(args.model):
        sys.exit(f'Model not found: {args.model}')

    library = args.library
    model = args.model
    level = int(args.level)

    # Video source
    video_source = args.video if args.video is not None else "0"
    if video_source == "0":
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        print("Using webcam")
    else:
        if not os.path.exists(video_source):
            sys.exit(f'Video file not found: {video_source}')
        cap = cv.VideoCapture(video_source)
        print(f"Playing video: {video_source}")

    if not cap.isOpened():
        sys.exit("Cannot open video source")

    # Output video
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    out = cv.VideoWriter('emotion_output.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps,
                         (frame_width, frame_height))

    # Initialize ASNN
    emotion_net = asnn('Electron')
    print(f'ASNN Version: {emotion_net.get_nn_version()}')
    print('Initializing neural network...')
    emotion_net.nn_init(library=library, model=model, level=level)
    print('Neural network ready.')

    print('Starting emotion detection. Press "q" to quit.')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        orig_frame = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        emotion_count = len(faces)
        print(f"Faces detected: {emotion_count}")

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]

            # Preprocess face for ASNN
            input_tensors = preprocess_face(face_gray)

            # Inference
            start_time = time.time()
            output_data = emotion_net.nn_inference(input_tensors,
                                                   platform='ONNX',
                                                   reorder='2 1 0',  # NCHW already done
                                                   output_tensor=1,
                                                   output_format=output_format.OUT_FORMAT_FLOAT32)
            inference_time = time.time() - start_time
            print(f'Inference time: {inference_time:.4f}s')

            # Postprocess
            emotion, confidence = postprocess_emotion(output_data)

            # Draw result
            draw_emotion(orig_frame, x, y, w, h, emotion, confidence)

        # Display count
        cv.putText(orig_frame, f'Faces: {emotion_count}', (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Show and save
        cv.imshow("Facial Emotion Recognition (ASNN)", orig_frame)
        out.write(orig_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("Done. Output saved as 'emotion_output.mp4'")
