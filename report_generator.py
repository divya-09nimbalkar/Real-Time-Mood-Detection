from collections import Counter
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Image as RLImage
import os

def generate_pdf_report(emotions):

    if len(emotions) == 0:
        print("No emotions detected.")
        return

    emotion_count = Counter(emotions)
    total = len(emotions)

    # Create bar chart
    plt.figure()
    plt.bar(emotion_count.keys(), emotion_count.values())
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("emotion_bar_chart.png")
    plt.close()

    # Create pie chart
    plt.figure()
    plt.pie(emotion_count.values(),
            labels=emotion_count.keys(),
            autopct='%1.1f%%')
    plt.title("Emotion Percentage")
    plt.tight_layout()
    plt.savefig("emotion_pie_chart.png")
    plt.close()

    # Create PDF
    doc = SimpleDocTemplate("Mood_Analysis_Report.pdf")
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Real-Time Mood Analysis Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Total Predictions: {total}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))

    dominant_emotion = emotion_count.most_common(1)[0][0]
    elements.append(Paragraph(f"Dominant Emotion: {dominant_emotion}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(RLImage("emotion_bar_chart.png", width=5*inch, height=3*inch))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(RLImage("emotion_pie_chart.png", width=5*inch, height=3*inch))

    doc.build(elements)

    # Cleanup images
    os.remove("emotion_bar_chart.png")
    os.remove("emotion_pie_chart.png")