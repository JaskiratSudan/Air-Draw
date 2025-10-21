import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

# resolution = (640, 480)
resolution = (1280, 720)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)
cap.set(3, resolution[0])
cap.set(4, resolution[1])

prev_frame_time = 0
new_frame_time = 0

detector = HandDetector(detectionCon=0.8, maxHands=1)

# Drawing configuration
toolbar_height = 110
color_options = [
    {"name": "Blue", "color": (255, 0, 0)},
    {"name": "Green", "color": (0, 255, 0)},
    {"name": "Red", "color": (0, 0, 255)},
    {"name": "Yellow", "color": (0, 255, 255)},
    {"name": "Purple", "color": (255, 0, 255)},
]
marker_options = [
    {"name": "Fine", "thickness": 5},
    {"name": "Medium", "thickness": 12},
    {"name": "Bold", "thickness": 20},
]

current_color_index = 0
current_marker_index = 1
eraser_thickness = 60

imgcanvas = np.zeros((resolution[1], resolution[0], 3), np.uint8)
xp, yp = 0, 0


def draw_toolbar(frame, selected_color, selected_marker):
    option_width = 140
    padding = 10

    # Draw color swatches
    for idx, option in enumerate(color_options):
        x1 = padding + idx * option_width
        x2 = x1 + option_width - padding
        y1, y2 = padding, toolbar_height // 2 - padding
        cv2.rectangle(frame, (x1, y1), (x2, y2), option["color"], cv2.FILLED)
        if idx == selected_color:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(
            frame,
            option["name"],
            (x1 + 5, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0) if sum(option["color"]) > 255 else (255, 255, 255),
            2,
        )

    # Draw marker thickness options
    base_y = toolbar_height // 2 + padding
    for idx, option in enumerate(marker_options):
        x1 = padding + idx * option_width
        x2 = x1 + option_width - padding
        y1, y2 = base_y, toolbar_height - padding
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 2)
        cv2.putText(
            frame,
            option["name"],
            (x1 + 10, y2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        thickness_preview_center = (x1 + (x2 - x1) // 2, base_y + (y2 - base_y) // 2)
        cv2.circle(
            frame,
            thickness_preview_center,
            max(2, option["thickness"] // 2),
            (255, 255, 255),
            cv2.FILLED,
        )
        if idx == selected_marker:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


while True:
    sucess, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    fps_text = f"FPS: {int(fps)}"
    cv2.putText(img, fps_text, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    draw_toolbar(img, current_color_index, current_marker_index)

    if hands:
        lmlist = hands[0]["lmList"]
        point_index = lmlist[8][0:2]
        point_middle = lmlist[12][0:2]

        fingers = detector.fingersUp(hands[0])

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if point_index[1] < toolbar_height:
                option_width = 140
                padding = 10
                # Determine selection index for colors
                if point_index[1] < toolbar_height // 2:
                    idx = (point_index[0] - padding) // option_width
                    if 0 <= idx < len(color_options):
                        current_color_index = idx
                else:
                    idx = (point_index[0] - padding) // option_width
                    if 0 <= idx < len(marker_options):
                        current_marker_index = idx
            else:
                cv2.rectangle(
                    img,
                    (point_index[0], point_index[1]),
                    (point_middle[0], point_middle[1]),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.rectangle(
                    imgcanvas,
                    (point_index[0], point_index[1]),
                    (point_middle[0], point_middle[1]),
                    (0, 0, 0),
                    eraser_thickness,
                    cv2.FILLED,
                )

        if fingers[1] and not fingers[2]:
            current_color = color_options[current_color_index]["color"]
            current_thickness = marker_options[current_marker_index]["thickness"]
            if xp == 0 and yp == 0:
                xp = point_index[0]
                yp = point_index[1]
            cv2.circle(img, point_index, 10, current_color, cv2.FILLED)
            cv2.line(
                imgcanvas,
                (xp, yp),
                (point_index[0], point_index[1]),
                current_color,
                current_thickness,
            )

            xp, yp = point_index[0], point_index[1]

        if not fingers[1] and not fingers[2]:
            xp, yp = 0, 0

    imgGray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgcanvas)

    cv2.imshow("Video", img)
    cv2.imshow("Canvas", imgInv)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
