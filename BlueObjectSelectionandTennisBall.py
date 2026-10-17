import cv2
import numpy as np
import math

camera = cv2.VideoCapture(0)

blue_min = np.array([100, 120, 50])
blue_max = np.array([135, 255, 255])

ball_min = np.array([18, 70, 50])
ball_max = np.array([45, 255, 255])

frame_data = {"image": None}

selected = False
last_x = 0
last_y = 0
last_box = None

ball_x = None
ball_y = None
ball_radius = None
ball_area = None
missing_frames = 0

kernel = np.ones((3, 3), np.uint8)


def select_blue(event, x, y, flags, data):
    global selected, last_x, last_y, last_box

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    image = data["image"]

    if image is None:
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blue_min, blue_max)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue

        if cv2.pointPolygonTest(
            contour, (float(x), float(y)), False
        ) >= 0:
            bx, by, bw, bh = cv2.boundingRect(contour)

            last_box = (bx, by, bw, bh)
            last_x = bx + bw // 2
            last_y = by + bh // 2
            selected = True
            break


cv2.namedWindow("Detection")
cv2.setMouseCallback("Detection", select_blue, frame_data)

while camera.isOpened():
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_data["image"] = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, blue_min, blue_max)
    blue_mask = cv2.morphologyEx(
        blue_mask, cv2.MORPH_OPEN, kernel
    )

    blue_contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blue_contours = [
        c for c in blue_contours if cv2.contourArea(c) > 100
    ]

    blue_center = None

    if not selected:
        cv2.putText(
            frame, "Click a blue object to track",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

        for contour in blue_contours:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (255, 0, 0), 1
            )

    elif last_box is not None:
        old_x, old_y, old_w, old_h = last_box

        nearest = None
        nearest_distance = float("inf")

        for contour in blue_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

            distance = math.hypot(cx - last_x, cy - last_y)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest = contour

        if nearest is not None and nearest_distance < 20:
            x, y, w, h = cv2.boundingRect(nearest)

            last_box = (
                int(0.8 * old_x + 0.2 * x),
                int(0.8 * old_y + 0.2 * y),
                int(0.8 * old_w + 0.2 * w),
                int(0.8 * old_h + 0.2 * h)
            )

        x, y, w, h = last_box
        last_x = x + w // 2
        last_y = y + h // 2
        blue_center = (last_x, last_y)

        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (255, 0, 0), 2
        )

        cv2.circle(frame, blue_center, 4, (255, 0, 0), -1)

        blue_area = None

        if nearest is not None and nearest_distance < 20:
            blue_area = cv2.contourArea(nearest)

        if blue_area is not None:
            cv2.putText(
                frame, f"Blue area: {blue_area:.0f} px^2",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 0, 0), 2
            )

        cv2.putText(
            frame, f"Box: x={x}, y={y}, w={w}, h={h}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 2
        )

        cv2.putText(
            frame, f"Blue center: {blue_center}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 2
        )

    ball_mask = cv2.inRange(hsv, ball_min, ball_max)
    ball_mask = cv2.morphologyEx(
        ball_mask, cv2.MORPH_OPEN, kernel
    )

    ball_contours, _ = cv2.findContours(
        ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in ball_contours:
        area = cv2.contourArea(contour)

        if area < 100 or area > 30000:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        roundness = 4 * math.pi * area / (perimeter * perimeter)

        if roundness < 0.65:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if min(w, h) == 0:
            continue

        ratio = w / h

        if ratio < 0.65 or ratio > 1.45:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        if radius < 5:
            continue

        candidates.append(
            (contour, cx, cy, radius, area, roundness)
        )

    best = None

    if candidates:
        if ball_x is None:
            best = max(candidates, key=lambda item: item[5])
        else:
            close_candidates = [
                item for item in candidates
                if math.hypot(
                    item[1] - ball_x, item[2] - ball_y
                ) < 100
            ]

            if close_candidates:
                best = min(
                    close_candidates,
                    key=lambda item:
                    math.hypot(
                        item[1] - ball_x, item[2] - ball_y
                    ) - item[5] * 10
                )

    ball_center = None

    if best is not None:
        _, cx, cy, radius, area, _ = best
        missing_frames = 0

        if ball_x is None:
            ball_x = cx
            ball_y = cy
            ball_radius = radius
            ball_area = area
        else:
            ball_x = 0.8 * ball_x + 0.2 * cx
            ball_y = 0.8 * ball_y + 0.2 * cy
            ball_radius = 0.8 * ball_radius + 0.2 * radius
            ball_area = 0.8 * ball_area + 0.2 * area

        cx = int(ball_x)
        cy = int(ball_y)
        radius = max(1, int(ball_radius))
        area = int(ball_area)

        ball_center = (cx, cy)

        cv2.circle(frame, ball_center, radius, (0, 255, 0), 2)
        cv2.circle(frame, ball_center, 4, (0, 0, 255), -1)

        cv2.putText(
            frame, f"Ball area: {area} px^2",
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Ball center: {ball_center}",
            (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Ball radius: {radius} px",
            (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 2
        )

    else:
        missing_frames += 1

        if missing_frames > 8:
            ball_x = None
            ball_y = None
            ball_radius = None
            ball_area = None

        cv2.putText(
            frame, "Tennis ball not detected",
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 200, 255), 2
        )

    if blue_center is not None and ball_center is not None:
        cv2.line(
            frame, blue_center, ball_center, (0, 255, 255), 2
        )

        dx = blue_center[0] - ball_center[0]
        dy = blue_center[1] - ball_center[1]

        cv2.putText(
            frame, f"Relative position: dx={dx}, dy={dy}",
            (10, 210), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 2
        )

    cv2.putText(
        frame, "R: Select another blue object | Q: Quit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 2
    )

    cv2.imshow("Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        selected = False
        last_x = 0
        last_y = 0
        last_box = None

camera.release()
cv2.destroyAllWindows()