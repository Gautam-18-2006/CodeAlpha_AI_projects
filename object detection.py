import cv2
import numpy as np
import os

# ---------------- PATH HANDLING ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(current_dir, filename)

# ---------------- SETTINGS ----------------
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 640)
cap.set(4, 480)

whT = 416          # Correct size for YOLOv3-tiny
confThr = 0.3      # Lower threshold (important)
nmsThr = 0.2

# ---------------- LOAD CLASS NAMES ----------------
with open(get_path("coco.names"), "r") as f:
    classes = f.read().strip().split("\n")

# ---------------- LOAD YOLO MODEL ----------------
net = cv2.dnn.readNet(
    get_path("yolov3.weights"),
    get_path("yolov3.cfg")
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ---------------- OBJECT DETECTION FUNCTION ----------------
def findObjects(outputs, img):
    hT, wT, _ = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            classScore = scores[classId]
            confidence = classScore * det[4]   # VERY IMPORTANT

            if confidence > confThr:
                w = int(det[2] * wT)
                h = int(det[3] * hT)
                x = int((det[0] * wT) - w / 2)
                y = int((det[1] * hT) - h / 2)

                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThr, nmsThr)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bbox[i]
            label = classes[classIds[i]]
            conf = int(confs[i] * 100)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label.upper()} {conf}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    if not success:
        break

    blob = cv2.dnn.blobFromImage(
        img,
        1 / 255,
        (whT, whT),
        [0, 0, 0],
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.imshow("YOLOv3-Tiny Object Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
