# Smart-Suveillance-System-

import cv2
import datetime

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Initialize first frame
first_frame = None

while True:
    # Read frame
    ret, frame = cap.read()
    text = "No Motion Detected"

    # Convert frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the first frame
    if first_frame is None:
        first_frame = gray
        continue

    # Compute difference between first frame and current frame
    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Motion Detected"

    # Put the text and timestamp on the frame
    cv2.putText(frame, f"Room Status: {text}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

    # Show the frame
    cv2.imshow("Security Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to break the loop
    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
