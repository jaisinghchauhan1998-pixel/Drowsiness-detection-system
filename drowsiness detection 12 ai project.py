import cv2
import dlib
import pyttsx3
import numpy as np
from scipy.spatial import distance
from pathlib import Path
from playsound3 import playsound


# -----------------------------------------
# INITIALIZE TEXT-TO-SPEECH
# -----------------------------------------

engine = pyttsx3.init()


# -----------------------------------------
# OPEN WEBCAM
# -----------------------------------------




# Check whether camera opened successfully
print("STEP 1: Starting camera...")

cap = cv2.VideoCapture(0)

print("STEP 2: VideoCapture created")

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("STEP 3: Camera successfully opened!")

ret, frame = cap.read()

print("STEP 4: Camera frame read completed")
print("ret =", ret)

if not ret or frame is None:
    print("ERROR: Could not read frame.")
    cap.release()
    exit()

print("STEP 5: Frame received!")
print("Frame shape:", frame.shape)
print("Frame type:", frame.dtype)

# -----------------------------------------
# FACE DETECTOR
# -----------------------------------------

face_detector = dlib.get_frontal_face_detector()


# -----------------------------------------
# LOAD 68-POINT FACIAL LANDMARK MODEL
# -----------------------------------------

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"

if not MODEL_PATH.exists():
    print("ERROR: Facial landmark model not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

print("Facial landmark model found!")

dlib_facelandmark = dlib.shape_predictor(str(MODEL_PATH))
# -----------------------------------------
# FUNCTION TO CALCULATE EYE ASPECT RATIO
# -----------------------------------------

def Detect_Eye(eye):

    poi_A = distance.euclidean(
        eye[1],
        eye[5]
    )

    poi_B = distance.euclidean(
        eye[2],
        eye[4]
    )

    poi_C = distance.euclidean(
        eye[0],
        eye[3]
    )

    aspect_ratio_Eye = (
        poi_A + poi_B
    ) / (2 * poi_C)

    return aspect_ratio_Eye


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------

while True:

    ret, frame = cap.read()

    # Check camera frame
    if not ret or frame is None:
        print("ERROR: Could not read camera frame.")
        break
    print("STEP A: Frame received")

    # Convert image to grayscale
    gray_scale = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Force Dlib-compatible 8-bit grayscale image
    gray_scale = np.asarray(
        gray_scale,
        dtype=np.uint8,
         order="C"
    )


    print("STEP B: Grayscale conversion completed")
    print("Gray shape:", gray_scale.shape)
    print("Gray dtype:", gray_scale.dtype)
    print("Gray dimensions:", gray_scale.ndim)
    print("Gray contiguous:", gray_scale.flags["C_CONTIGUOUS"])

    # Detect faces
    print("STEP 6: About to detect faces...")

    faces = face_detector(gray_scale)

    print("STEP 7: Face detection completed!")
    print("Faces detected:", len(faces))


# Process every detected face
for face in faces:

    # Find 68 facial landmarks
    face_landmarks = dlib_facelandmark(
        gray_scale,
        face
    )


    leftEye = []
    rightEye = []


        # ---------------------------------
        # EYE 1
        # LANDMARKS 42 - 47
        # ---------------------------------

    for n in range(42, 48):

            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y

            rightEye.append((x, y))

            next_point = n + 1

            if n == 47:
                next_point = 42

            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y

            cv2.line(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                1
            )


        # ---------------------------------
        # EYE 2
        # LANDMARKS 36 - 41
        # ---------------------------------

    for n in range(36, 42):

            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y

            leftEye.append((x, y))

            next_point = n + 1

            if n == 41:
                next_point = 36

            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y

            cv2.line(
                frame,
                (x, y),
                (x2, y2),
                (255, 255, 0),
                1
            )

    # =========================================
# CALCULATE EYE ASPECT RATIO
# =========================================

    right_Eye = Detect_Eye(rightEye)
    left_Eye = Detect_Eye(leftEye)

    Eye_Rat = (left_Eye + right_Eye) / 2
    Eye_Rat = round(Eye_Rat, 2)

    print("Eye Ratio:", Eye_Rat)


    # ---------------------------------
    # SHOW CAMERA
    # ---------------------------------

    cv2.imshow(
        "Drowsiness Detector",
        frame
    )


    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break


# -----------------------------------------
# CLEANUP
# -----------------------------------------

cap.release()
cv2.destroyAllWindows()