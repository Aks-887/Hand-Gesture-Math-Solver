import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import numpy as np
from PIL import Image
import pytesseract
import ast
import operator
import time
from collections import deque

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------- CONFIG --------------------
MODEL_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0
MAX_HANDS = 1
SMOOTHING_WINDOW = 4
DRAW_THICKNESS = 6
OCR_THICKNESS = 10
GESTURE_HOLD_FRAMES = 8
ACTION_COOLDOWN_FRAMES = 12
PADDING = 30

# -------------------- SAFE EVAL --------------------
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def safe_eval(expr: str):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):  # compatibility
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)

# -------------------- HAND CONNECTIONS --------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_custom_landmarks(image, landmarks):
    h, w, _ = image.shape
    for s, e in HAND_CONNECTIONS:
        p1 = (int(landmarks[s].x * w), int(landmarks[s].y * h))
        p2 = (int(landmarks[e].x * w), int(landmarks[e].y * h))
        cv2.line(image, p1, p2, (255, 255, 255), 2)
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

# -------------------- GESTURES --------------------
def fingers_up_count(landmarks):
    # Simplified upright-camera logic
    tips_and_joints = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]
    count = 0
    for tip, joint in tips_and_joints[1:]:  # skip thumb in this simple logic
        if landmarks[tip].y < landmarks[joint].y:
            count += 1
    return count

def recognize_gesture(landmarks):
    index_tip = landmarks[8]
    index_pip = landmarks[6]

    # draw: only index finger up
    idx_up = index_tip.y < index_pip.y
    mid_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    if idx_up and not mid_up and not ring_up and not pinky_up:
        return "draw"

    # calculate: all fingers folded (fist)
    tips_and_joints = [(4, 2), (8, 6), (12, 10), (16, 14), (20, 18)]
    is_fist = all(landmarks[tip].y > landmarks[joint].y for tip, joint in tips_and_joints)
    if is_fist:
        return "solve"

    # undo: index + middle up
    if idx_up and mid_up and not ring_up and not pinky_up:
        return "undo"

    return "idle"

# -------------------- HELPERS --------------------
def smooth_point(history, point):
    history.append(point)
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

def draw_strokes(image, strokes, color=(0, 255, 0), thickness=DRAW_THICKNESS):
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(image, stroke[i - 1], stroke[i], color, thickness)

def build_ocr_image(shape, strokes):
    h, w = shape[:2]
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    all_points = []

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        all_points.extend(stroke)
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1], stroke[i], 0, OCR_THICKNESS)

    if not all_points:
        return None

    pts = np.array(all_points)
    min_x = max(int(np.min(pts[:, 0])) - PADDING, 0)
    max_x = min(int(np.max(pts[:, 0])) + PADDING, w)
    min_y = max(int(np.min(pts[:, 1])) - PADDING, 0)
    max_y = min(int(np.max(pts[:, 1])) + PADDING, h)

    roi = canvas[min_y:max_y, min_x:max_x]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, roi = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    return roi

def solve_expression_from_strokes(frame_shape, strokes):
    roi = build_ocr_image(frame_shape, strokes)
    if roi is None:
        return "", "", None

    pil_img = Image.fromarray(roi)
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-*/().%'
    expr = pytesseract.image_to_string(pil_img, config=config).strip()
    expr = expr.replace("x", "*").replace("X", "*").replace("÷", "/")

    try:
        if expr:
            result = str(safe_eval(expr))
        else:
            result = ""
    except Exception:
        result = "Invalid"

    return expr, result, roi

# -------------------- MEDIAPIPE SETUP --------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_hands=MAX_HANDS
)
detector = vision.HandLandmarker.create_from_options(options)

# -------------------- APP STATE --------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

strokes = []
current_stroke = []
point_history = deque(maxlen=SMOOTHING_WINDOW)

gesture_buffer = deque(maxlen=GESTURE_HOLD_FRAMES)
cooldown = 0

last_expr = ""
last_result = ""
ocr_preview = None

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    current_gesture = "idle"

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        draw_custom_landmarks(display, hand_landmarks)
        current_gesture = recognize_gesture(hand_landmarks)
        gesture_buffer.append(current_gesture)

        h, w, _ = frame.shape
        index_tip = hand_landmarks[8]
        raw_point = (int(index_tip.x * w), int(index_tip.y * h))
        smooth = smooth_point(point_history, raw_point)

        stable_gesture = None
        if len(gesture_buffer) == GESTURE_HOLD_FRAMES and len(set(gesture_buffer)) == 1:
            stable_gesture = gesture_buffer[-1]

        if cooldown > 0:
            cooldown -= 1
            stable_gesture = None

        if stable_gesture == "draw":
            current_stroke.append(smooth)

        elif stable_gesture == "undo":
            if current_stroke:
                current_stroke = []
            elif strokes:
                strokes.pop()
            cooldown = ACTION_COOLDOWN_FRAMES

        elif stable_gesture == "solve":
            if current_stroke:
                strokes.append(current_stroke.copy())
                current_stroke = []

            expr, ans, preview = solve_expression_from_strokes(frame.shape, strokes)
            if expr:
                last_expr = expr
                last_result = ans
                ocr_preview = preview
            cooldown = ACTION_COOLDOWN_FRAMES

        else:
            if current_stroke:
                strokes.append(current_stroke.copy())
                current_stroke = []

    else:
        gesture_buffer.append("idle")
        point_history.clear()
        if current_stroke:
            strokes.append(current_stroke.copy())
            current_stroke = []

    # draw saved strokes
    draw_strokes(display, strokes)
    if len(current_stroke) > 1:
        for i in range(1, len(current_stroke)):
            cv2.line(display, current_stroke[i - 1], current_stroke[i], (0, 255, 0), DRAW_THICKNESS)

    # UI
    cv2.rectangle(display, (0, 0), (display.shape[1], 95), (30, 30, 30), -1)
    cv2.putText(display, f"Mode: {current_gesture.upper()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, "Keys: S=Solve  U=Undo  C=Clear  Q=Quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if last_expr:
        cv2.putText(display, f"{last_expr} = {last_result}", (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    if ocr_preview is not None:
        preview_bgr = cv2.cvtColor(ocr_preview, cv2.COLOR_GRAY2BGR)
        ph, pw = preview_bgr.shape[:2]
        target_w = 220
        scale = target_w / pw
        preview_bgr = cv2.resize(preview_bgr, (target_w, int(ph * scale)))
        ph, pw = preview_bgr.shape[:2]
        y1 = min(110, display.shape[0] - ph - 10)
        x1 = display.shape[1] - pw - 10
        display[y1:y1+ph, x1:x1+pw] = preview_bgr
        cv2.rectangle(display, (x1, y1), (x1+pw, y1+ph), (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Math Solver", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        strokes.clear()
        current_stroke.clear()
        last_expr = ""
        last_result = ""
        ocr_preview = None
    elif key == ord('u'):
        if current_stroke:
            current_stroke.clear()
        elif strokes:
            strokes.pop()
    elif key == ord('s'):
        if current_stroke:
            strokes.append(current_stroke.copy())
            current_stroke.clear()
        expr, ans, preview = solve_expression_from_strokes(frame.shape, strokes)
        if expr:
            last_expr = expr
            last_result = ans
            ocr_preview = preview

cap.release()
cv2.destroyAllWindows()