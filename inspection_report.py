"""Generate a portable PDF inspection report from JSON-friendly results."""

from pathlib import Path


def create_pdf_report(output_path, title, detections, summary=None):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as error:
        raise RuntimeError("PDF reports require ReportLab. Install it with: pip install reportlab") from error
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#1769aa"))), Spacer(1, 12)]
    if summary:
        story.append(Paragraph(f"Summary: {summary}", styles["BodyText"]))
        story.append(Spacer(1, 12))
    rows = [["Type", "Severity", "Priority", "Confidence", "Latitude", "Longitude"]]
    rows.extend([[item.get("damage_type", "N/A"), item.get("severity", "N/A"), item.get("priority_level", "N/A"), f"{float(item.get('confidence', 0)):.1%}", item.get("latitude", "N/A"), item.get("longitude", "N/A")] for item in detections])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1769aa")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e2ec")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fb")]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(table)
    SimpleDocTemplate(str(output_path), pagesize=letter).build(story)
    return output_path


def report_payload(title, detections, summary=None):
    """Return a serializable report payload for API/export clients."""
    return {"title": title, "summary": summary or {}, "detection_count": len(detections), "detections": detections}
