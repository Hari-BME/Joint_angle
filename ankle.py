import mediapipe as mp
import cv2
import math
import streamlit as st
import tempfile
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit UI
st.title("Pose Estimation and Joint Angle Measurement")

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded file into memory
    image1 = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Correct the image format
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Function to calculate angle between three points
    def calculate_angle(a, b, c):
        ab = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        bc = math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
        ac = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)
        angle_rad = math.acos((ab**2 + bc**2 - ac**2) / (2 * ab * bc))
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    # Process the image and detect keypoints
    with mp_pose.Pose() as pose:
        results = pose.process(image)  # No need to convert to RGB again

    if results.pose_landmarks:
        st.write("Keypoints Detected:")
        st.image(image, caption="Detected Keypoints", use_column_width=True)

        # Extract keypoints for shoulder, hip, and knee
        landmarks = results.pose_landmarks.landmark
        shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.shape[1],
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.shape[0])
        hip = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image.shape[1],
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image.shape[0])
        knee = (landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image.shape[0])

        # Draw point A, B, and C with the vertical line between A and C
        # Draw point A, B, and C with the vertical line between A and C
        cv2.circle(image, (int(hip[0]), int(hip[1])), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(shoulder[0]), int(shoulder[1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(knee[0]), int(knee[1])), 5, (0, 0, 255), -1)

        # Draw lines between points
        cv2.line(image, (int(shoulder[0]), int(shoulder[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 2)
        cv2.line(image, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 255, 0), 2)
        
        # Calculate angles
        angle_shoulder_hip_knee = calculate_angle(shoulder, hip, knee)

        # Display keypoints and angles
        st.write("Left Shoulder:", shoulder)
        st.write("Left Hip:", hip)
        st.write("Left Knee:", knee)
    
        st.write("Angle (Shoulder-Hip-Knee): {:.2f} degrees".format(angle_shoulder_hip_knee))
     # Display the image using cv2.imshow
    cv2.imshow("Detected Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    st.error("No keypoints detected in the image. Please upload a different image.")
