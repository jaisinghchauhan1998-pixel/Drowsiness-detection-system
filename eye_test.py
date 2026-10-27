import cv2
import dlib
import numpy as np
from pathlib import Path
from scipy.spatial import distance


# -----------------------------------------
# LOAD 68-POINT FACIAL LANDMARK MODEL
# -----------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"

if not MODEL_PATH.exists():
    print("ERROR: Facial landmark model not found!")
    print(MODEL_PATH)
    exit()

print("Facial landmark model found!")

face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor(str(MODEL_PATH))


# -----------------------------------------
# EYE ASPECT RATIO FUNCTION
# -----------------------------------------

def Detect_Eye(eye):

    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])

    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)

    return aspect_ratio_Eye


# -----------------------------------------
# CAMERA
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

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.asarray(
        gray,
        dtype=np.uint8,
        order="C"
    )

    faces = face_detector(gray)

    for face in faces:

        landmarks = dlib_facelandmark(gray, face)

        # ---------------------------------
        # RIGHT EYE: LANDMARKS 36-41
        # ---------------------------------

        rightEye = []

        for n in range(36, 42):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            rightEye.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ---------------------------------
        # LEFT EYE: LANDMARKS 42-47
        # ---------------------------------

        leftEye = []

        for n in range(42, 48):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            leftEye.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ---------------------------------
        # CALCULATE EYE RATIOS
        # ---------------------------------

        right_Eye = Detect_Eye(rightEye)

        left_Eye = Detect_Eye(leftEye)

        Eye_Rat = (left_Eye + right_Eye) / 2

        Eye_Rat = round(Eye_Rat, 2)


        # ---------------------------------
        # DISPLAY EYE RATIO
        # ---------------------------------

        cv2.putText(
            frame,
            f"Eye Ratio: {Eye_Rat}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ---------------------------------
    # SHOW CAMERA
    # ---------------------------------

    cv2.imshow(
        "Eye Detection Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# -----------------------------------------
# CLEANUP
# -----------------------------------------

cap.release()
cv2.destroyAllWindows()