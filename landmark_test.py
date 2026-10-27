import cv2
import dlib
import numpy as np
from pathlib import Path


# -----------------------------------------
# LOAD 68-POINT FACIAL LANDMARK MODEL
# -----------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"

if not MODEL_PATH.exists():
    print("ERROR: Facial landmark model not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

print("Facial landmark model found!")

face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor(str(MODEL_PATH))


# -----------------------------------------
# OPEN CAMERA
# -----------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("Press Q in the camera window to quit.")


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("ERROR: Could not read camera frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Make sure Dlib receives compatible data
    gray = np.asarray(
        gray,
        dtype=np.uint8,
        order="C"
    )

    # Detect faces
    faces = face_detector(gray)

    # Process every detected face
    for face in faces:

        # Get 68 facial landmarks
        landmarks = dlib_facelandmark(gray, face)

        # Draw all 68 landmark points
        for n in range(68):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    # Show camera
    cv2.imshow(
        "68 Facial Landmarks Test",
        frame
    )

    # Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# -----------------------------------------
# CLEANUP
# -----------------------------------------

cap.release()
cv2.destroyAllWindows()