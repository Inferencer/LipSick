import os
import cv2

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_and_crop_frame(video_path, frame_number, crop_to_face=True):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    
    output_dir = "utils/temp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
    cap.release()

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_detected = len(faces) > 0

    # Overlay text on the frame
    frame_with_text = overlay_text(frame, frame_number, face_detected=face_detected)
    # Convert RGB back to BGR before saving
    frame_with_text = cv2.cvtColor(frame_with_text, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, frame_with_text)
    return output_path


def overlay_text(frame, frame_number, face_detected=True):
    #print("Debugging overlay_text:")
    #print(f"Frame number: {frame_number}")
    #print(f"Face detected: {face_detected}")

    # Define font and text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2  # Increase thickness for bold effect
    ref_text = f"Reference Frame: {frame_number}"
    ref_text_color = (0, 255, 0)  # Green for reference frame text in BGR
    no_face_text = "NO FACE DETECTED"
    no_face_text_color = (255, 0, 0)  # Red for no face detected text in BGR

    # Get text sizes
    ref_text_size, _ = cv2.getTextSize(ref_text, font, font_scale, font_thickness)
    no_face_text_size, _ = cv2.getTextSize(no_face_text, font, font_scale, font_thickness)

    # Calculate text positions
    ref_text_x = (frame.shape[1] - ref_text_size[0]) // 2
    ref_text_y = (frame.shape[0] // 4) + (ref_text_size[1] // 2)  # Adjust position halfway between middle and top

    no_face_text_x = (frame.shape[1] - no_face_text_size[0]) // 2
    no_face_text_y = ref_text_y + ref_text_size[1] + 20  # Below reference text with some padding

    # Overlay reference text on frame with black background
    ref_text_bg_x1 = ref_text_x - 10
    ref_text_bg_y1 = ref_text_y - ref_text_size[1] - 10
    ref_text_bg_x2 = ref_text_x + ref_text_size[0] + 10
    ref_text_bg_y2 = ref_text_y + 10
    cv2.rectangle(frame, (ref_text_bg_x1, ref_text_bg_y1), (ref_text_bg_x2, ref_text_bg_y2), (0, 0, 0), -1)
    cv2.putText(frame, ref_text, (ref_text_x, ref_text_y), font, font_scale, ref_text_color, font_thickness)

    # Overlay "NO FACE DETECTED" text if no face is detected
    if not face_detected:
        no_face_text_bg_x1 = no_face_text_x - 10
        no_face_text_bg_y1 = no_face_text_y - no_face_text_size[1] - 10
        no_face_text_bg_x2 = no_face_text_x + no_face_text_size[0] + 10
        no_face_text_bg_y2 = no_face_text_y + 10
        cv2.rectangle(frame, (no_face_text_bg_x1, no_face_text_bg_y1), (no_face_text_bg_x2, no_face_text_bg_y2), (0, 0, 0), -1)
        cv2.putText(frame, no_face_text, (no_face_text_x, no_face_text_y), font, font_scale, no_face_text_color, font_thickness)

    return frame


def delete_existing_reference_frames():
    # Directory to store reference frames
    ref_frames_dir = "utils/temp"
    
    # Check if the directory exists, if not create it
    if not os.path.exists(ref_frames_dir):
        return  # No existing frames to delete
    
    # Delete existing reference frames if any
    existing_frames = [f for f in os.listdir(ref_frames_dir) if os.path.isfile(os.path.join(ref_frames_dir, f))]
    for frame in existing_frames:
        os.remove(os.path.join(ref_frames_dir, frame))
