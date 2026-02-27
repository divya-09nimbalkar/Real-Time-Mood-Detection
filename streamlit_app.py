import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import matplotlib.pyplot as plt
from report_generator import generate_pdf_report
import tempfile

st.title("Video Mood Analysis System")

model = load_model("emotion_model.h5")

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    emotion_log = []
    frame_count = 0

    st.write("Processing video... Please wait.")

    progress_bar = st.progress(0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 5th frame
        if frame_count % 5 != 0:
            continue

        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(min(progress, 100))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float32") / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            prediction = model.predict(roi, verbose=0)
            max_index = np.argmax(prediction)
            emotion = emotion_labels[max_index]

            emotion_log.append(emotion)

    cap.release()

    if len(emotion_log) == 0:
        st.error("No faces detected in video.")
    else:
        emotion_counts = Counter(emotion_log)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)

        st.subheader("Analysis Summary")
        st.write(f"Total Frames Analyzed: {len(emotion_log)}")
        st.write(f"Dominant Emotion: {dominant_emotion}")

        st.subheader("Emotion Frequency Distribution")

        fig, ax = plt.subplots()
        ax.bar(emotion_counts.keys(), emotion_counts.values())
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Frequency")
        ax.set_title("Emotion Distribution")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        filename = generate_pdf_report(emotion_log)

        with open(filename, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=filename,
                mime="application/pdf"
            )

        st.success("Video Analysis Completed Successfully!")