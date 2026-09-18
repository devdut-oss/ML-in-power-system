"""
build_all_pdfs.py
=================
Builds PDF reports for the project using reportlab.
Combines all markdown documentation into a single PDF with watermark.

Generates:
- PowerSystemProtection_ML_Report.pdf (original docs)
- REAL_DATASET_REPORT.pdf (new real dataset docs)
"""

import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame,
                                Paragraph, Spacer, PageBreak, Preformatted,
                                Table, TableStyle, Image, HRFlowable)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
OUT_DIR = ROOT

PAGE_W, PAGE_H = A4

# ---------------------------------------------------------------- styles ---
ss = getSampleStyleSheet()
STY = {
    'title': ParagraphStyle('title', parent=ss['Title'], fontSize=20,
                            spaceAfter=6, textColor=colors.HexColor('#0b2a52')),
    'h1': ParagraphStyle('h1', parent=ss['Heading1'], fontSize=15,
                         spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor('#0b2a52')),
    'h2': ParagraphStyle('h2', parent=ss['Heading2'], fontSize=12.5,
                         spaceBefore=10, spaceAfter=4,
                         textColor=colors.HexColor('#14508f')),
    'h3': ParagraphStyle('h3', parent=ss['Heading3'], fontSize=11,
                         spaceBefore=8, spaceAfter=3,
                         textColor=colors.HexColor('#14508f')),
    'body': ParagraphStyle('body', parent=ss['BodyText'], fontSize=9.5,
                           leading=13.5, spaceAfter=4),
    'quote': ParagraphStyle('quote', parent=ss['BodyText'], fontSize=9.5,
                            leading=13.5, leftIndent=14, spaceAfter=4,
                            textColor=colors.HexColor('#444444'),
                            borderPadding=4),
    'bullet': ParagraphStyle('bullet', parent=ss['BodyText'], fontSize=9.5,
                             leading=13, leftIndent=14, bulletIndent=4,
                             spaceAfter=2),
    'code': ParagraphStyle('code', parent=ss['Code'], fontSize=7.3,
                           leading=9.2, leftIndent=8,
                           backColor=colors.HexColor('#f4f4f0'),
                           borderPadding=5, spaceAfter=6),
    'caption': ParagraphStyle('caption', parent=ss['BodyText'], fontSize=8.5,
                              alignment=TA_CENTER,
                              textColor=colors.HexColor('#555555')),
}


# ------------------------------------------------------- inline md -> html --
def inline(md):
    t = (md.replace('&', '&').replace('<', '<').replace('>', '>'))
    t = re.sub(r'`([^`]+)`',
               r'<font face="Courier" size="8.5" color="#8b2252">\1</font>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'<i>\1</i>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<u>\1</u>', t)   # strip links
    return t


def md_table(lines):
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            continue                                   # separator row
        rows.append([Paragraph(inline(c), STY['body']) for c in cells])
    if not rows:
        return None
    ncols = max(len(r) for r in rows)
    for r in rows:
        r.extend([''] * (ncols - len(r)))
    avail = PAGE_W - 40 * mm
    tbl = Table(rows, colWidths=[avail / ncols] * ncols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#b9c4d4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    return tbl


def md_to_flowables(md_text):
    flows = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        # fenced code block
        if s.startswith('```'):
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block.append(lines[i])
                i += 1
            i += 1
            flows.append(Preformatted('\n'.join(block), STY['code']))
            continue
        # frontmatter/HR
        if s in ('---', '***') and (i == 0 or not lines[i-1].strip()):
            flows.append(HRFlowable(width='100%', thickness=0.6,
                                    color=colors.HexColor('#b9c4d4'),
                                    spaceBefore=6, spaceAfter=6))
            i += 1
            continue
        # table
        if s.startswith('|') and i + 1 < len(lines) and \
                re.match(r'^\s*\|[\s:|-]+\|?\s*$', lines[i + 1]):
            tl = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                tl.append(lines[i])
                i += 1
            t = md_table(tl)
            if t:
                flows.append(Spacer(1, 3))
                flows.append(t)
                flows.append(Spacer(1, 5))
            continue
        # headings
        m = re.match(r'^(#{1,4})\s+(.*)', s)
        if m:
            level = len(m.group(1))
            key = {1: 'h1', 2: 'h2', 3: 'h3', 4: 'h3'}[level]
            flows.append(Paragraph(inline(m.group(2)), STY[key]))
            i += 1
            continue
        # blockquote
        if s.startswith('>'):
            q = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                q.append(lines[i].strip().lstrip('>').strip())
                i += 1
            flows.append(Paragraph(inline(' '.join(q)), STY['quote']))
            continue
        # bullets / numbered (merge soft-wrapped continuation lines)
        def take_item(first):
            item = [first]
            nonlocal_i = [i + 1]
            while nonlocal_i[0] < len(lines):
                nxt = lines[nonlocal_i[0]].strip()
                if (not nxt or nxt.startswith(('#', '|', '```', '>', '- ', '* '))
                        or re.match(r'^\d+[.)]\s', nxt) or nxt in ('---', '***')):
                    break
                item.append(nxt)
                nonlocal_i[0] += 1
            return ' '.join(item), nonlocal_i[0]

        m = re.match(r'^\s*[-*]\s+(.*)', ln)
        if m:
            text, i = take_item(m.group(1))
            flows.append(Paragraph(inline(text), STY['bullet'], bulletText='•'))
            continue
        m = re.match(r'^\s*(\d+)[.)]\s+(.*)', ln)
        if m:
            text, i = take_item(m.group(2))
            flows.append(Paragraph(inline(text), STY['bullet'],
                                   bulletText=f'{m.group(1)}.'))
            continue
        # blank
        if not s:
            i += 1
            continue
        # normal paragraph (merge soft-wrapped lines)
        para = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(('#', '|', '```', '>', '- ', '* '))
                    or re.match(r'^\d+[.)]\s', nxt) or nxt in ('---', '***')):
                break
            para.append(nxt)
            i += 1
        flows.append(Paragraph(inline(' '.join(para)), STY['body']))
    return flows


# ------------------------------------------------------------- watermark ---
def paint_page(canv, doc):
    """Watermark + footer, drawn INTO the page content stream."""
    canv.saveState()
    # --- watermark: small, italic, centred, light grey ---
    canv.setFont('Times-Italic', 22)
    canv.setFillColor(colors.Color(0.55, 0.55, 0.58, alpha=0.28))
    canv.translate(PAGE_W / 2, PAGE_H / 2)
    canv.rotate(30)
    canv.drawCentredString(0, 0, "DEV's")
    canv.restoreState()
    # --- footer ---
    canv.saveState()
    canv.setFont('Helvetica', 7.5)
    canv.setFillColor(colors.HexColor('#888888'))
    canv.drawCentredString(PAGE_W / 2, 10 * mm,
                           f'Power System Protection using ML  --  page {doc.page}')
    canv.restoreState()


def build_pdf(out_path, title, subtitle, sections, cover_lines=None):
    """Build a PDF from markdown sections."""
    doc = BaseDocTemplate(out_path, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=16 * mm, bottomMargin=16 * mm,
                          title=title, author="DEV's")
    frame = Frame(20 * mm, 16 * mm, PAGE_W - 40 * mm, PAGE_H - 32 * mm, id='main')
    doc.addPageTemplates([PageTemplate(id='all', frames=[frame], onPage=paint_page)])

    story = []

    # ---- cover ----
    story.append(Spacer(1, 60 * mm))
    story.append(Paragraph(title, STY['title']))
    if subtitle:
        story.append(Spacer(1, 6))
        story.append(Paragraph(subtitle, STY['h3']))
    if cover_lines:
        story.append(Spacer(1, 10))
        for line in cover_lines:
            story.append(Paragraph(line, STY['body']))
    story.append(PageBreak())

    # ---- sections ----
    for label, path in sections:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path, encoding='utf-8') as f:
            content = f.read()
        story.extend(md_to_flowables(content))
        story.append(PageBreak())

    doc.build(story)
    print(f'PDF written: {out_path}')


def main():
    print("=" * 60)
    print("Building PDF Reports")
    print("=" * 60)

    # ---- Report 1: Original Project Docs ----
    print("\n[1/2] Building main project report...")
    sections_main = [
        ('README.md', os.path.join(ROOT, 'README.md')),
        ('PROJECT_PROPOSAL.md', os.path.join(DOCS, 'PROJECT_PROPOSAL.md')),
        ('METHODOLOGY.md', os.path.join(DOCS, 'METHODOLOGY.md')),
        ('STUDY_GUIDE.md', os.path.join(DOCS, 'STUDY_GUIDE.md')),
        ('LITERATURE_SURVEY.md', os.path.join(DOCS, 'LITERATURE_SURVEY.md')),
    ]
    build_pdf(
        os.path.join(OUT_DIR, 'PowerSystemProtection_ML_Report.pdf'),
        'Power System Protection<br/>using Machine Learning',
        'Fault Classification, Zone Discrimination, Relay Selection and IDMT '
        'Characteristic Emulation on a 132 kV Transmission Feeder',
        sections_main,
        cover_lines=[
            'Major Project — Electrical Engineering (Power Systems)',
            'Simulated + Real Dataset from MATLAB/Simulink',
            'Random Forest / MLP / SVM / Decision Tree',
        ]
    )

    # ---- Report 2: Real Dataset Summary ----
    print("\n[2/2] Building real dataset report...")
    sections_real = [
        ('PROJECT_SUMMARY.md', os.path.join(ROOT, 'PROJECT_SUMMARY.md')),
        ('REAL_DATASET_SUMMARY.md', os.path.join(ROOT, 'REAL_DATASET_SUMMARY.md')),
    ]
    build_pdf(
        os.path.join(OUT_DIR, 'REAL_DATASET_REPORT.pdf'),
        'Real Dataset ML Protection<br/>132 kV Transmission Feeder',
        'MATLAB/Simulink Time-Domain Data → Phasor Features → '
        'Random Forest Classification & Regression',
        sections_real,
        cover_lines=[
            '45 Simulink .mat files (v7.3 HDF5)',
            '45 fault cases + 100 healthy cases = 145 rows',
            'Tasks: Fault Classification | Zone Detection | IDMT Trip Time',
            'CV Accuracy: 100% / 100% | MAE: 6–40 ms',
        ]
    )

    print("\n" + "=" * 60)
    print("All PDFs generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()