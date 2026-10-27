import cv2
import dlib
import time
import numpy as np

face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera opened successfully.")

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("ERROR: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = np.asarray(
        gray,
        dtype=np.uint8,
        order="C"
    )

    start_time = time.perf_counter()

    faces = face_detector(gray, 0)

    elapsed = time.perf_counter() - start_time

    print(
        f"Dlib detection: {elapsed:.3f} seconds | "
        f"Faces: {len(faces)}"
    )

    cv2.imshow("Dlib Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()