#!/usr/bin/env python3
"""report/compile_pdf.py — Manuscript PDF Compiler.

Compiles report/research_paper.md into a publication-style PDF report/paper.pdf
with formatted headings, mathematical expressions, embedded PNG figures,
and structured data tables using ReportLab.

Outputs:
  - report/paper.pdf
"""
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

REPO_ROOT = Path(__file__).resolve().parents[1]

def clean_md_text(text: str) -> str:
    """Clean markdown formatting for ReportLab Paragraphs."""
    # Convert bold **text** to <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Convert italic *text* to <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Convert math LaTeX symbols like $TSR$ or \text{...}
    text = re.sub(r'\$(.*?)\$', r'<b><i>\1</i></b>', text)
    text = text.replace(r'\text{TSR}', 'TSR').replace(r'\text{CSS}', 'CSS').replace(r'\text{OAR}', 'OAR')
    text = text.replace(r'\text{PRIDE}', 'PRIDE').replace(r'\Delta', 'Δ').replace(r'\approx', '≈')
    text = text.replace(r'\le', '≤').replace(r'\ge', '≥').replace(r'\to', '→').replace(r'\times', '×')
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Restore bold/italic tags
    text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
    text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
    return text

def parse_markdown_to_story(md_path: Path):
    """Parse research_paper.md into ReportLab flowables."""
    md_text = md_path.read_text()
    lines = md_text.splitlines()

    styles = getSampleStyleSheet()
    
    # Custom styles for paper manuscript
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=colors.HexColor('#1a2530'), alignment=TA_LEFT, spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'H1', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=14, leading=18,
        textColor=colors.HexColor('#2c3e50'), spaceBefore=14, spaceAfter=8, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=12, leading=15,
        textColor=colors.HexColor('#34495e'), spaceBefore=10, spaceAfter=6, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontName='Times-Roman', fontSize=10, leading=14,
        textColor=colors.HexColor('#2c3e50'), alignment=TA_JUSTIFY, spaceAfter=6
    )
    abstract_style = ParagraphStyle(
        'Abstract', parent=styles['Normal'],
        fontName='Times-Italic', fontSize=9.5, leading=13.5,
        textColor=colors.HexColor('#34495e'), alignment=TA_JUSTIFY, leftIndent=15, rightIndent=15, spaceAfter=12
    )
    caption_style = ParagraphStyle(
        'Caption', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8.5, leading=11,
        textColor=colors.HexColor('#555555'), alignment=TA_CENTER, spaceBefore=4, spaceAfter=10
    )

    story = []
    in_abstract = False
    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Document Title (# Title)
        if line.startswith("# "):
            title_text = clean_md_text(line[2:])
            story.append(Paragraph(title_text, title_style))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2c3e50'), spaceAfter=12))
            i += 1
            continue

        # Section Headings (## Heading)
        if line.startswith("## "):
            story.append(Paragraph(clean_md_text(line[3:]), h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#bdc3c7'), spaceAfter=6))
            i += 1
            continue

        # Sub-Section Headings (### Heading or #### Heading)
        if line.startswith("### ") or line.startswith("#### "):
            h_text = line.lstrip("#").strip()
            story.append(Paragraph(clean_md_text(h_text), h2_style))
            i += 1
            continue

        # Images (![Alt](filename.png))
        if line.startswith("!["):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if m:
                alt, img_name = m.group(1), m.group(2)
                img_path = REPO_ROOT / "report" / img_name
                if img_path.exists():
                    img = Image(str(img_path), width=480, height=220)
                    story.append(Spacer(1, 6))
                    story.append(img)
                    story.append(Paragraph(f"Figure: {clean_md_text(alt)}", caption_style))
                i += 1
                continue

        # Tables (| Col 1 | Col 2 |)
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            
            # Process markdown table lines
            matrix = []
            for tl in table_lines:
                if "---" in tl or ":---" in tl:
                    continue
                cells = [clean_md_text(c.strip()) for c in tl.split("|")[1:-1]]
                if cells:
                    matrix.append([Paragraph(c, body_style) for c in cells])
            
            if matrix:
                t = Table(matrix)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 8))
            continue

        # Abstract section
        if "**Abstract**" in line or line.startswith("Abstract"):
            in_abstract = True
            line = line.replace("**Abstract**", "").strip()
            if line:
                story.append(Paragraph(f"<b>Abstract</b> — {clean_md_text(line)}", abstract_style))
            i += 1
            continue

        # Regular Body Paragraphs
        p_text = clean_md_text(line)
        if p_text:
            story.append(Paragraph(p_text, body_style))
        i += 1

    return story

def main():
    md_file = REPO_ROOT / "report" / "research_paper.md"
    pdf_file = REPO_ROOT / "report" / "paper.pdf"

    if not md_file.exists():
        print(f"Error: {md_file} not found.")
        sys.exit(1)

    print(f"[compile_pdf] Reading {md_file}...", flush=True)
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=letter,
        leftMargin=54, rightMargin=54,
        topMargin=54, bottomMargin=54
    )

    story = parse_markdown_to_story(md_file)
    print(f"[compile_pdf] Building PDF with {len(story)} flowable elements...", flush=True)
    doc.build(story)
    print(f"wrote {pdf_file}")

if __name__ == "__main__":
    main()
