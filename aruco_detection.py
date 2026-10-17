import cv2
import numpy as np
import math
print("NumPy is working:", np.__version__)
MARKER_LENGTH=0.05
CAMERA_INDEX=0
dictionary_names = {
    "4x4": cv2.aruco.DICT_4X4_50,
    "5x5": cv2.aruco.DICT_5X5_50,
    "6x6": cv2.aruco.DICT_6X6_50
}

detectors = {}

for name, dictionary_id in dictionary_names.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    detectors[name] = cv2.aruco.ArucoDetector(dictionary, parameters)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

half_length = MARKER_LENGTH / 2

object_points = np.array([
    [-half_length, half_length, 0],
    [half_length, half_length, 0],
    [half_length, -half_length, 0],
    [-half_length, -half_length, 0]
], dtype=np.float32)

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    height, width = frame.shape[:2]

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion_coefficients = np.zeros((5, 1), dtype=np.float64)

    for dictionary_name, detector in detectors.items():
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is None:
            continue

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            image_points = corners[i].reshape(4, 2).astype(np.float32)

            pose_found, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not pose_found:
                continue

            X = float(tvec[0][0])
            Y = float(tvec[1][0])
            Z = float(tvec[2][0])

            X_cm = X * 100
            Y_cm = Y * 100
            Z_cm = Z * 100

            distance_cm = math.sqrt(X**2 + Y**2 + Z**2) * 100

            rotation_matrix, _ = cv2.Rodrigues(rvec)

            sy = math.sqrt(
                rotation_matrix[0, 0] ** 2
                + rotation_matrix[1, 0] ** 2
            )

            singular = sy < 1e-6

            if not singular:
                roll = math.atan2(
                    rotation_matrix[2, 1],
                    rotation_matrix[2, 2]
                )
                pitch = math.atan2(
                    -rotation_matrix[2, 0],
                    sy
                )
                yaw = math.atan2(
                    rotation_matrix[1, 0],
                    rotation_matrix[0, 0]
                )
            else:
                roll = math.atan2(
                    -rotation_matrix[1, 2],
                    rotation_matrix[1, 1]
                )
                pitch = math.atan2(
                    -rotation_matrix[2, 0],
                    sy
                )
                yaw = 0

            roll_deg = math.degrees(roll)
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                distortion_coefficients,
                rvec,
                tvec,
                MARKER_LENGTH * 0.5
            )

            top_left = corners[i][0][0]
            text_x = max(10, int(top_left[0]))
            text_y = max(25, int(top_left[1]) - 10)

            cv2.putText(
                frame,
                f"{dictionary_name} ID: {marker_id}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"X:{X_cm:.1f} Y:{Y_cm:.1f} Z:{Z_cm:.1f} cm",
                (text_x, text_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"R:{roll_deg:.1f} P:{pitch_deg:.1f} Yaw:{yaw_deg:.1f} deg",
                (text_x, text_y + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Distance: {distance_cm:.1f} cm",
                (text_x, text_y + 71),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("ArUco pose estimation stopped.")