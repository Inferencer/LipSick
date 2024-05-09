import os
import cv2
import numpy as np
import dlib
from utils.data_processing import compute_crop_radius

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

def extract_frames_from_video(video_path):
    videoCapture = cv2.VideoCapture(video_path)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_list = []
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frame_list.append(frame)
    return frame_list

def load_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if not faces:
        raise ValueError("No faces found in the frame.")
    shape = landmark_predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def calculate_crop_radius_statistics(video_path):
    frames = extract_frames_from_video(video_path)
    landmark_data = [load_landmarks(frame) for frame in frames]

    crop_radii = []
    for i in range(len(landmark_data) - 5):
        landmark_data_clip = np.array(landmark_data[i:i+5])
        if landmark_data_clip.shape != (5, 68, 2):
            continue
        crop_flag, crop_radius = compute_crop_radius(frames[0].shape[:2], landmark_data_clip)
        if crop_flag:
            crop_radii.append(crop_radius)
    
    if not crop_radii:
        raise ValueError("No valid crop radius values found.")
    
    lowest = min(crop_radii)
    highest = max(crop_radii)
    average = int(np.mean(crop_radii))
    most_common = int(np.median(crop_radii))

    # Printing detailed statistics
    print("────────────────────────────────────────────")
    print("Computing Crop Radius Statistics...")
    print("────────────────────────────────────────────")
    print(f"Done! \nLowest Crop Radius = {lowest}\nHighest Crop Radius = {highest}\nAverage Crop Radius = {average}\nMost Common Crop Radius = {most_common}")
    print("────────────────────────────────────────────")

    return lowest, highest, average, most_common
