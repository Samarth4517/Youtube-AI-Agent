"""
Generates a downloadable PDF of a summary (or any generated text block)
using reportlab. Returns raw PDF bytes suitable for st.download_button.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT


def build_pdf(title: str, video_url: str, sections: dict) -> bytes:
    """
    sections: dict of {heading: body_text} rendered in order, e.g.
        {"Concise Summary": "...", "Key Points": "..."}
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, topMargin=0.75 * inch, bottomMargin=0.75 * inch
    )
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["BodyText"], alignment=TA_LEFT, leading=16
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], textColor="#666666", fontSize=9
    )

    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Source: {video_url}", meta_style),
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style
        ),
        Spacer(1, 12),
    ]

    for heading, body in sections.items():
        if not body:
            continue
        story.append(Paragraph(heading, heading_style))
        # Preserve line breaks from the LLM output as separate paragraphs.
        for line in body.split("\n"):
            if line.strip():
                # Escape characters that would break reportlab's mini-HTML parser.
                safe_line = (
                    line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                story.append(Paragraph(safe_line, body_style))

    doc.build(story)
    return buffer.getvalue()
