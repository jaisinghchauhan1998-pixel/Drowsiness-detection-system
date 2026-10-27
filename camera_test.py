import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera opened successfully.")
print("Press Q to close the camera window.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()