import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- LOAD MODEL ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "emotion_model.h5")

model = load_model(model_path, compile=False)
print("Model Loaded Successfully!")

# ---------------- EMOTIONS ---------------- #

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# ---------------- FACE DETECTOR ---------------- #

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error loading Haarcascade.")
    exit()

# ---------------- SESSION STORAGE ---------------- #

session_emotions = []

# ---------------- CAMERA ---------------- #

cap = cv2.VideoCapture(0)
print("Session started... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        session_emotions.append(emotion)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    cv2.imshow("Real-Time Mood Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Session ended. Generating report...")

# ---------------- REPORT GENERATION ---------------- #

if len(session_emotions) == 0:
    print("No emotions detected.")
    exit()

emotion_counts = Counter(session_emotions)

# Create Plot
plt.figure()
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.xticks(rotation=45)
plt.title("Emotion Distribution")
plt.tight_layout()

plot_path = os.path.join(BASE_DIR, "emotion_plot.png")
plt.savefig(plot_path)
plt.close()

# Create PDF
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdf_path = os.path.join(BASE_DIR, f"Mood_Report_{timestamp}.pdf")

doc = SimpleDocTemplate(pdf_path, pagesize=A4)
elements = []

styles = getSampleStyleSheet()

elements.append(Paragraph("<b>Real-Time Mood Analysis Report</b>", styles['Title']))
elements.append(Spacer(1, 0.5*inch))

elements.append(Paragraph(f"Session Date: {datetime.now()}", styles['Normal']))
elements.append(Spacer(1, 0.5*inch))

# Table Data
data = [["Emotion", "Frequency"]]
for emotion, count in emotion_counts.items():
    data.append([emotion, count])

table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.grey),
    ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
    ('GRID', (0,0), (-1,-1), 1, colors.black)
]))

elements.append(table)
elements.append(Spacer(1, 0.5*inch))

# Add Plot Image
elements.append(Image(plot_path, width=5*inch, height=3*inch))

doc.build(elements)

print("Report Generated Successfully!")
print("Saved at:", pdf_path)