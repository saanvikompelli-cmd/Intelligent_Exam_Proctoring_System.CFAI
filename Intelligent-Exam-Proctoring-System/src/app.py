import cv2

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

suspicious_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    warning = ""

    # No face detected
    if len(faces) == 0:
        suspicious_count += 1
        warning = "ALERT: No Face Detected!"

    # Multiple faces detected
    elif len(faces) > 1:
        suspicious_count += 1
        warning = "WARNING: Multiple Faces Detected!"

    # Draw rectangles around faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Face center
        center_x = x + w // 2

        # Head movement detection
        if center_x < 200:
            suspicious_count += 1
            warning = "Looking Left Frequently!"

        elif center_x > 450:
            suspicious_count += 1
            warning = "Looking Right Frequently!"

    # Project Title
    cv2.putText(
        frame,
        "Intelligent Exam Proctoring System",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Suspicious activity counter
    cv2.putText(
        frame,
        f"Suspicious Activities: {suspicious_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Warning display
    cv2.putText(
        frame,
        warning,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Face count
    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Show window
    cv2.imshow(
        "AI Exam Proctoring System",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()