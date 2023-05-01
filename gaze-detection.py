import cv2
import dlib
import math

# Initialize webcam and dlib's face detector and shape predictor
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\minha\Downloads\shape_predictor_68_face_landmarks.dat")

while True:
    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each face
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Extract the left and right eye landmarks
        left_eye = []
        right_eye = []
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))

        # Calculate the midpoint of each eye
        left_midpoint = ((left_eye[0][0]+left_eye[3][0])//2, (left_eye[0][1]+left_eye[3][1])//2)
        right_midpoint = ((right_eye[0][0]+right_eye[3][0])//2, (right_eye[0][1]+right_eye[3][1])//2)
        radius = 100

        # Draw a circle around the midpoints of the eyes
        cv2.circle(frame, left_midpoint, 2, (0, 0, 255), thickness=5)
        cv2.circle(frame, right_midpoint, 2, (0, 0, 255), thickness=5)


        # Calculate the angle between the midpoints of the eyes and the horizontal axis
        delta_x = right_midpoint[0] - left_midpoint[0]
        delta_y = right_midpoint[1] - left_midpoint[1]
        angle = math.atan2(delta_y, delta_x) * 180 / math.pi

        # Print the angle on the frame
        cv2.putText(frame, f"Gaze Angle: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        if angle>-2 and angle<2:
            cv2.putText(frame, "Looking Straight", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        if angle >2:
            cv2.putText(frame, "Looking Left", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        if angle<-2:
            cv2.putText(frame, "Looking Right", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
    

    cv2.imshow("Gaze Detection", frame)

    if cv2.waitKey(1)==ord('q'):
        break


# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
