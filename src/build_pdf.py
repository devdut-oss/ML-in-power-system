"""
build_pdf.py
=============
Combines all project markdown docs into one PDF with a flattened "DEV's"
watermark drawn into every page's content stream (small, italic, centred).
Because the watermark is part of the page content itself (not an annotation
or overlay object), it cannot be deleted the way stamp/annotation watermarks
can.

Also embeds all 7 result figures as an appendix.

Run:  python src/build_pdf.py
Out:  PowerSystemProtection_ML_Report.pdf  (project root)
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
FIGS = os.path.join(ROOT, 'results', 'figures')
OUT = os.path.join(ROOT, 'PowerSystemProtection_ML_Report.pdf')

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
    t = (md.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
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
    """Watermark + header/footer, drawn INTO the page content stream."""
    canv.saveState()
    # --- the DEV's watermark: small, italic, centred, light grey ---
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
                           f'Power System Protection using ML  —  '
                           f'page {doc.page}')
    canv.restoreState()


# ------------------------------------------------------------------ build --
def main():
    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=16 * mm, bottomMargin=16 * mm,
                          title='Power System Protection using ML',
                          author="DEV's")
    frame = Frame(20 * mm, 16 * mm, PAGE_W - 40 * mm, PAGE_H - 32 * mm,
                  id='main')
    doc.addPageTemplates([PageTemplate(id='all', frames=[frame],
                                       onPage=paint_page)])

    story = []
    # ---- cover ----
    story.append(Spacer(1, 60 * mm))
    story.append(Paragraph('Power System Protection<br/>using Machine Learning',
                           STY['title']))
    story.append(Spacer(1, 6))
    story.append(Paragraph('Fault Classification, Zone Discrimination, Relay '
                           'Selection and IDMT Characteristic Emulation on a '
                           '132 kV Transmission Feeder', STY['h3']))
    story.append(Spacer(1, 10))
    story.append(Paragraph('Major Project — Electrical Engineering '
                           '(Power Systems)', STY['body']))
    story.append(Paragraph('Combined project report: proposal, methodology, '
                           'study guide, literature survey and results.',
                           STY['body']))
    story.append(PageBreak())

    sections = [
        ('README.md', os.path.join(ROOT, 'README.md')),
        ('PROJECT_PROPOSAL.md', os.path.join(DOCS, 'PROJECT_PROPOSAL.md')),
        ('METHODOLOGY.md', os.path.join(DOCS, 'METHODOLOGY.md')),
        ('STUDY_GUIDE.md', os.path.join(DOCS, 'STUDY_GUIDE.md')),
        ('LITERATURE_SURVEY.md', os.path.join(DOCS, 'LITERATURE_SURVEY.md')),
    ]
    for label, path in sections:
        with open(path, encoding='utf-8') as f:
            story.extend(md_to_flowables(f.read()))
        story.append(PageBreak())

    # ---- appendix: figures ----
    story.append(Paragraph('Appendix — Result Figures', STY['h1']))
    captions = {
        'fig1_system_diagram.png': 'Fig. 1 — Single-line diagram of the study system',
        'fig2_fault_currents.png': 'Fig. 2 — Fault current at R1 vs fault location',
        'fig3_idmt_curves.png': 'Fig. 3 — Coordinated IEC-SI IDMT characteristics',
        'fig4_confusion_matrices.png': 'Fig. 4 — Confusion matrices (test set)',
        'fig5_regression.png': 'Fig. 5 — Location & trip-time regression',
        'fig6_feature_importance.png': 'Fig. 6 — Random-Forest feature importance',
        'fig7_idmt_ml_vs_analytical.png': 'Fig. 7 — ML-learned vs analytical IDMT curve',
    }
    avail_w = PAGE_W - 44 * mm
    for fname, cap in captions.items():
        p = os.path.join(FIGS, fname)
        img = Image(p)
        scale = min(avail_w / img.imageWidth, 1.0)
        # keep two shorter figures per page where possible
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        if img.drawHeight > 120 * mm:
            k = 120 * mm / img.drawHeight
            img.drawWidth *= k
            img.drawHeight *= k
        story.append(Spacer(1, 6))
        story.append(img)
        story.append(Paragraph(cap, STY['caption']))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f'PDF written: {OUT}')


if __name__ == '__main__':
    main()
