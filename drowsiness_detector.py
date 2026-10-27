import cv2
import dlib
import numpy as np
import threading
from pathlib import Path
from scipy.spatial import distance
from playsound3 import playsound
import threading

ALARM_PATH = Path(__file__).resolve().parent / "alarm.mp3"
alarm_lock = threading.Lock()
alarm_playing = False

def play_alarm():

    global alarm_playing

    try:
        print("🔊 Alarm started")
        playsound(str(ALARM_PATH))

    except Exception as e:
        print("Alarm error:", e)

    finally:
        with alarm_lock:
            alarm_playing = False

        print("🔇 Alarm finished")


# =========================================
# SETTINGS
# =========================================

EAR_THRESHOLD = 0.20
CLOSED_FRAMES_REQUIRED = 15


# =========================================
# LOAD FACIAL LANDMARK MODEL
# =========================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "shape_predictor_68_face_landmarks.dat"
)

if not MODEL_PATH.exists():
    print("ERROR: Facial landmark model not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

print("Facial landmark model found!")


# =========================================
# DLIB
# =========================================

face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor(
    str(MODEL_PATH)
)


# =========================================
# EYE ASPECT RATIO
# =========================================

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


# =========================================
# CAMERA
# =========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("Press Q in the camera window to quit.")


# =========================================
# DROWSINESS COUNTER
# =========================================

closed_frames = 0
alarm_playing = False


# =========================================
# MAIN LOOP
# =========================================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:

        print("ERROR: Could not read camera frame.")
        break


    # -------------------------------------
    # GRAYSCALE
    # -------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.asarray(
        gray,
        dtype=np.uint8,
        order="C"
    )


    # -------------------------------------
    # FACE DETECTION
    # -------------------------------------

    faces = face_detector(gray)


    # -------------------------------------
    # PROCESS FACE
    # -------------------------------------

    for face in faces:

        landmarks = dlib_facelandmark(
            gray,
            face
        )


        # ---------------------------------
        # RIGHT EYE
        # ---------------------------------

        rightEye = []

        for n in range(36, 42):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            rightEye.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )


        # ---------------------------------
        # LEFT EYE
        # ---------------------------------

        leftEye = []

        for n in range(42, 48):

            x = landmarks.part(n).x
            y = landmarks.part(n).y

            leftEye.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )


        # ---------------------------------
        # CALCULATE EAR
        # ---------------------------------

        right_Eye = Detect_Eye(rightEye)

        left_Eye = Detect_Eye(leftEye)

        Eye_Rat = (
            left_Eye + right_Eye
        ) / 2

        Eye_Rat = round(
            Eye_Rat,
            2
        )


        # ---------------------------------
        # CHECK EYES
        # ---------------------------------

        if Eye_Rat < EAR_THRESHOLD:

            closed_frames += 1

        else:

            closed_frames = 0


        # ---------------------------------
        # STATUS
        # ---------------------------------

        if closed_frames >= CLOSED_FRAMES_REQUIRED:

           status = "DROWSINESS DETECTED"


           with alarm_lock:

               if not alarm_playing:

                   alarm_playing = True

                   alarm_thread = threading.Thread(
                       target=play_alarm,
                       daemon=True
                   ) 

                   alarm_thread.start()

        else:

            status = "AWAKE"
            


        # ---------------------------------
        # DISPLAY EAR
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
        # DISPLAY COUNTER
        # ---------------------------------

        cv2.putText(
            frame,
            f"Closed Frames: {closed_frames}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # ---------------------------------
        # DISPLAY STATUS
        # ---------------------------------

        cv2.putText(
            frame,
            status,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255) if status == "DROWSINESS DETECTED"
            else (0, 255, 0),
            2
        )


    # =====================================
    # SHOW CAMERA
    # =====================================

    cv2.imshow(
        "Drowsiness Detection Test",
        frame
    )


    # =====================================
    # QUIT
    # =====================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================
# CLEANUP
# =========================================

cap.release()

cv2.destroyAllWindows()