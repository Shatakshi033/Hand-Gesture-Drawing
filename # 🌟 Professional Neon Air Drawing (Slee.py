import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime

# ---------------------------
# Configuration / Constants
# ---------------------------
BTN_Y = 20
BTN_H = 60
BTN_W = 160
BTN_SPACING = 20
BTN_START_X = 30

# Neon colors for buttons (BGR)
NEON = {
    "CLEAR": (10, 220, 220),
    "BLUE":  (200, 120, 20),
    "GREEN": (40, 220, 40),
    "RED":   (20, 20, 255),
    "YELLOW":(20, 240, 240),
    "SAVE":  (180, 30, 200)
}

# Drawing colors (BGR)
DRAW_COLORS = {
    0: (255, 0, 0),     # Blue
    1: (0, 255, 0),     # Green
    2: (0, 0, 255),     # Red
    3: (0, 255, 255)    # Yellow
}

# ---------------------------
# Button setup
# ---------------------------
labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW", "SAVE"]
buttons = []
x = BTN_START_X
for lbl in labels:
    buttons.append((lbl, x, BTN_Y, x + BTN_W, BTN_Y + BTN_H))
    x += BTN_W + BTN_SPACING

# ---------------------------
# Drawing data structures
# ---------------------------
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colorIndex = 0  # default color: blue

# ---------------------------
# Mediapipe hands init
# ---------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# ---------------------------
# Utility functions
# ---------------------------
def draw_neon_button(img, x1, y1, x2, y2, text, color, selected=False, hover=False):
    """Draw a neon-styled button."""
    overlay = img.copy()
    glow_color = tuple(int(c) for c in color)
    max_glow = 6
    for i in range(max_glow, 0, -1):
        alpha = 0.08 * (max_glow - i + 1)
        thickness = i * 2
        cv2.rectangle(overlay, (x1 - i, y1 - i), (x2 + i, y2 + i), glow_color, thickness)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Fill if selected
    if selected:
        fill_overlay = img.copy()
        cv2.rectangle(fill_overlay, (x1+2, y1+2), (x2-2, y2-2), color, -1)
        cv2.addWeighted(fill_overlay, 0.12, img, 0.88, 0, img)

    # Hover highlight
    if hover:
        hov = img.copy()
        cv2.rectangle(hov, (x1+3, y1+3), (x2-3, y2-3), (255,255,255), -1)
        cv2.addWeighted(hov, 0.06, img, 0.94, 0, img)

    # Main border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

    # Text
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
    tx = x1 + (x2 - x1 - text_size[0]) // 2
    ty = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(img, text, (tx, ty), font, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

def save_paint(canvas):
    """Save the canvas with timestamp."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"HandDrawing_{now}.png"
    cv2.imwrite(fname, canvas)
    print(f"[Saved] {fname}")

# ---------------------------
# Main loop
# ---------------------------
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Unable to access webcam")

paintWindow = None
prevPinch = False
pinch_threshold = 25

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Initialize canvas
    if paintWindow is None or paintWindow.shape[:2] != frame.shape[:2]:
        paintWindow = np.ones_like(frame) * 255
        # Draw buttons on canvas
        for lbl, x1, y1, x2, y2 in buttons:
            draw_neon_button(paintWindow, x1, y1, x2, y2, lbl, NEON[lbl])

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hover_button = None
    # Draw UI buttons on live frame
    for lbl, x1, y1, x2, y2 in buttons:
        selected = (lbl != "CLEAR" and lbl != "SAVE") and (
            (lbl=="BLUE" and colorIndex==0) or 
            (lbl=="GREEN" and colorIndex==1) or 
            (lbl=="RED" and colorIndex==2) or 
            (lbl=="YELLOW" and colorIndex==3)
        )
        draw_neon_button(frame, x1, y1, x2, y2, lbl, NEON[lbl], selected=selected, hover=False)

    if result.multi_hand_landmarks:
        for handLMS in result.multi_hand_landmarks:
            lm8 = handLMS.landmark[8]   # index fingertip
            lm4 = handLMS.landmark[4]   # thumb tip
            x, y = int(lm8.x * w), int(lm8.y * h)
            thumb_y = int(lm4.y * h)

            cv2.circle(frame, (x, y), 6, (0,0,0), -1)

            is_pinch = abs(thumb_y - y) < pinch_threshold

            # Hover detection
            for lbl, x1, y1, x2, y2 in buttons:
                if x1 < x < x2 and y1 < y < y2:
                    draw_neon_button(frame, x1, y1, x2, y2, lbl, NEON[lbl], selected=(lbl!="CLEAR" and lbl!="SAVE" and ((lbl=="BLUE" and colorIndex==0) or (lbl=="GREEN" and colorIndex==1) or (lbl=="RED" and colorIndex==2) or (lbl=="YELLOW" and colorIndex==3))), hover=True)
                    hover_button = lbl

            # Button click
            if is_pinch and not prevPinch and hover_button is not None:
                lbl = hover_button
                if lbl == "CLEAR":
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    blue_index = green_index = red_index = yellow_index = 0
                    paintWindow[:] = 255
                    for lb, x1, y1, x2, y2 in buttons:
                        draw_neon_button(paintWindow, x1, y1, x2, y2, lb, NEON[lb])
                    print("[Action] CLEAR")
                elif lbl == "SAVE":
                    save_paint(paintWindow)
                else:
                    colorIndex = {"BLUE":0,"GREEN":1,"RED":2,"YELLOW":3}[lbl]
                    print(f"[Action] Selected color: {lbl}")

                # start new stroke group
                bpoints.append(deque(maxlen=1024)); blue_index += 1
                gpoints.append(deque(maxlen=1024)); green_index += 1
                rpoints.append(deque(maxlen=1024)); red_index += 1
                ypoints.append(deque(maxlen=1024)); yellow_index += 1

            # Draw points
            if not (y <= (BTN_Y + BTN_H) and hover_button is not None and is_pinch):
                if colorIndex == 0: bpoints[blue_index].appendleft((x, y))
                elif colorIndex == 1: gpoints[green_index].appendleft((x, y))
                elif colorIndex == 2: rpoints[red_index].appendleft((x, y))
                elif colorIndex == 3: ypoints[yellow_index].appendleft((x, y))

            prevPinch = is_pinch
            mpDraw.draw_landmarks(frame, handLMS, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(100,100,100), thickness=1, circle_radius=2),
                                  mpDraw.DrawingSpec(color=(50,50,50), thickness=1))

    else:
        prevPinch = False

    # Draw strokes
    for i, pts in enumerate([bpoints, gpoints, rpoints, ypoints]):
        for dq in pts:
            for k in range(1, len(dq)):
                if dq[k-1] and dq[k]:
                    cv2.line(frame, dq[k-1], dq[k], DRAW_COLORS[i], 4)
                    cv2.line(paintWindow, dq[k-1], dq[k], DRAW_COLORS[i], 4)

    # Show frames
    cv2.imshow("NeonPaint - Camera", frame)
    cv2.imshow("NeonPaint - Canvas (SAVE will store this)", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # ESC or 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
