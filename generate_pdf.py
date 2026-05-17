#!/usr/bin/env python3
"""
Generate Project_Report.pdf from Project_Report.md with all figures inline.
Uses fpdf2 for PDF generation with a custom markdown parser.
"""

import os
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos

BASE_DIR  = "/Users/xiaoqin/Documents/claude code test/WESAD"
MD_FILE   = os.path.join(BASE_DIR, "Project_Report.md")
OUT_FILE  = os.path.join(BASE_DIR, "Project_Report.pdf")
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
MONO_PATH = "/System/Library/Fonts/Monaco.ttf"

# ── colour palette ────────────────────────────────────────────────────────────
C_BLACK   = (0,   0,   0)
C_DARK    = (30,  30,  30)
C_HEADING = (20,  60, 120)
C_RULE    = (180, 180, 180)
C_CODE_BG = (245, 245, 245)
C_CODE_FG = (50,  50,  50)
C_TABLE_H = (220, 230, 245)
C_TABLE_R = (255, 255, 255)
C_TABLE_A = (248, 249, 252)
C_CAPTION = (80,  80,  80)


def sanitize(text):
    """Replace non-latin-1 characters with ASCII equivalents."""
    replacements = {
        '\u2014': '--',   # em dash
        '\u2013': '-',    # en dash
        '\u2192': '->',   # arrow right
        '\u00d7': 'x',    # multiplication sign
        '\u2265': '>=',   # >=
        '\u2264': '<=',   # <=
        '\u2248': '~',    # approximately
        '\u00b0': ' deg', # degree
        '\u00b9': '1',    # superscript 1
        '\u00b2': '2',    # superscript 2
        '\u00b3': '3',    # superscript 3
        '\u207b': '-',    # superscript minus
        '\u2019': "'",    # right single quote
        '\u2018': "'",    # left single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
        '\u2022': '-',    # bullet
        '\u00e9': 'e',    # e acute
        '\u03b1': 'alpha','\u03b2': 'beta', '\u03b3': 'gamma',
        '\u00b5': 'mu',   # micro sign
        '\u03c3': 'sigma',
        '\u00b1': '+/-',  # plus-minus
        '\u2026': '...',  # ellipsis
        '\u00a0': ' ',    # non-breaking space
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Final fallback: drop remaining non-latin-1 chars
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(format='A4', unit='mm')
        self.set_margins(20, 20, 20)
        self.set_auto_page_break(auto=True, margin=22)
        # page width available
        self.pw = self.w - self.l_margin - self.r_margin

    # ── header / footer ───────────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(*C_CAPTION)
        self.cell(0, 6, 'Emotion Recognition from PPG -- CSC 491/591', align='L',
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*C_RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(*C_CAPTION)
        self.cell(0, 6, f'Page {self.page_no()}', align='C')

    # ── helpers ───────────────────────────────────────────────────────────────
    def rule(self):
        self.set_draw_color(*C_RULE)
        self.ln(1)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def write_paragraph(self, text, size=10, bold=False, italic=False,
                        color=C_DARK, indent=0, line_height=5.5):
        """Render a paragraph supporting inline **bold** and *italic* markup."""
        style = ('B' if bold else '') + ('I' if italic else '')
        self.set_font('Helvetica', style, size)
        self.set_text_color(*color)
        if indent:
            self.set_x(self.l_margin + indent)
        # Strip leading/trailing whitespace
        text = text.strip()
        if not text:
            return
        # Parse inline bold/italic
        segments = _parse_inline(text)
        for seg_text, seg_bold, seg_italic in segments:
            s = ('B' if (bold or seg_bold) else '') + ('I' if (italic or seg_italic) else '')
            self.set_font('Helvetica', s, size)
            self.set_text_color(*color)
            self.write(line_height, sanitize(seg_text))
        self.ln(line_height * 0.4)

    def write_code_block(self, lines):
        """Render a monospaced code block with a light background."""
        self.set_fill_color(*C_CODE_BG)
        self.set_draw_color(*C_RULE)
        self.ln(2)
        x0, y0 = self.get_x(), self.get_y()
        # Draw background rect (we'll extend it after writing text)
        self.set_font('Courier', '', 8)
        self.set_text_color(*C_CODE_FG)
        # Pad with 3 mm on each side
        PAD = 3
        inner_w = self.pw
        for i, line in enumerate(lines):
            if i == 0:
                # Draw full-height box — approximate height
                pass
            self.set_x(self.l_margin + PAD)
            self.cell(inner_w - PAD, 4.5, sanitize(line), fill=(i % 2 == 0),
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        # Draw border around block
        y1 = self.get_y()
        self.rect(self.l_margin, y0 - 1, inner_w, y1 - y0 + 2)
        self.ln(3)

    def write_table(self, headers, rows, col_widths=None):
        """Render a markdown-style table."""
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [self.pw / n_cols] * n_cols
        else:
            # normalise to available width
            total = sum(col_widths)
            col_widths = [w / total * self.pw for w in col_widths]

        ROW_H = 5.5
        FONT_SIZE = 8.5

        # Force a page break before drawing if the full table won't fit,
        # so the header is never stranded at the bottom of a page without rows.
        est_h = ROW_H * (len(rows) + 1) + 8
        if self.get_y() + est_h > self.h - self.b_margin:
            self.add_page()

        def _draw_row(cells, fill_color, bold=False, font_size=FONT_SIZE):
            self.set_fill_color(*fill_color)
            self.set_draw_color(*C_RULE)
            self.set_font('Helvetica', 'B' if bold else '', font_size)
            self.set_text_color(*C_DARK)
            max_lines = 1
            # measure line-wrapped heights first
            cell_texts = []
            for i, cell in enumerate(cells):
                cell = str(cell).strip().strip('*').strip('`')
                cell_texts.append(cell)
                chars_per_line = max(1, int(col_widths[i] / (font_size * 0.37)))
                lines_needed = max(1, (len(cell) + chars_per_line - 1) // chars_per_line)
                max_lines = max(max_lines, lines_needed)
            h = ROW_H * max_lines
            # check page break
            if self.get_y() + h > self.h - self.b_margin:
                self.add_page()
                # re-draw header on new page
                _draw_row(headers, C_TABLE_H, bold=True)
            for i, cell in enumerate(cell_texts):
                align = 'C' if (bold or i == 0) else 'L'
                # multi_cell for wrapping
                x = self.get_x()
                y = self.get_y()
                self.multi_cell(col_widths[i], ROW_H, sanitize(cell),
                                border=1, align=align, fill=True,
                                new_x=XPos.RIGHT, new_y=YPos.TOP)
                self.set_xy(self.get_x(), y)
            self.ln(h)

        self.ln(2)
        _draw_row(headers, C_TABLE_H, bold=True)
        for ri, row in enumerate(rows):
            fill = C_TABLE_R if ri % 2 == 0 else C_TABLE_A
            _draw_row(row, fill)
        self.ln(3)

    def insert_figure(self, img_path, caption):
        """Insert a figure centered on the page with a caption below."""
        if not os.path.exists(img_path):
            self.write_paragraph(f'[Figure not found: {img_path}]', italic=True, color=C_CAPTION)
            return
        from PIL import Image as PILImage
        with PILImage.open(img_path) as im:
            w_px, h_px = im.size
        # Max width = full page width; max height = 130 mm
        MAX_W = self.pw
        MAX_H = 130
        aspect = h_px / w_px
        img_w = MAX_W
        img_h = img_w * aspect
        if img_h > MAX_H:
            img_h = MAX_H
            img_w = img_h / aspect
        # Check if figure + caption fits, else new page
        needed = img_h + 14  # caption ~14 mm
        if self.get_y() + needed > self.h - self.b_margin:
            self.add_page()
        x_offset = (self.pw - img_w) / 2 + self.l_margin
        self.image(img_path, x=x_offset, y=None, w=img_w, h=img_h)
        # Caption
        self.ln(2)
        self.set_font('Helvetica', 'I', 8.5)
        self.set_text_color(*C_CAPTION)
        self.multi_cell(self.pw, 4.5, sanitize(caption), align='C',
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)


# ── inline markup parser ──────────────────────────────────────────────────────
def _parse_inline(text):
    """
    Parse inline **bold**, *italic*, `code` and return list of
    (text, is_bold, is_italic) tuples.
    Strip markdown image/link syntax entirely.
    """
    # Remove image syntax ![alt](url) entirely
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # Remove link syntax [text](url) → just text
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)

    segments = []
    # Combined pattern
    pat = re.compile(r'\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`')
    last = 0
    for m in pat.finditer(text):
        if m.start() > last:
            segments.append((text[last:m.start()], False, False))
        if m.group(1):
            segments.append((m.group(1), True, False))
        elif m.group(2):
            segments.append((m.group(2), False, True))
        elif m.group(3):
            segments.append((m.group(3), False, False))  # code inline as normal
        last = m.end()
    if last < len(text):
        segments.append((text[last:], False, False))
    return segments


# ── markdown table parser ─────────────────────────────────────────────────────
def _parse_md_table(lines):
    """
    Given a list of raw markdown table lines, return (headers, rows, col_widths).
    col_widths is proportional to max content length in each column.
    """
    # strip separator line (---|---)
    data_lines = [l for l in lines if not re.match(r'^\s*\|?[-:\s|]+\|?\s*$', l)]
    if len(data_lines) < 1:
        return None, None, None

    def parse_row(line):
        line = line.strip()
        if line.startswith('|'): line = line[1:]
        if line.endswith('|'): line = line[:-1]
        return [c.strip() for c in line.split('|')]

    rows_raw = [parse_row(l) for l in data_lines]
    n_cols = max(len(r) for r in rows_raw)
    # pad short rows
    rows_raw = [r + [''] * (n_cols - len(r)) for r in rows_raw]

    headers = rows_raw[0]
    rows    = rows_raw[1:]

    # compute proportional column widths from content
    col_max = [max(len(str(r[i])) for r in rows_raw) for i in range(n_cols)]
    col_max = [max(c, 5) for c in col_max]

    return headers, rows, col_max


# ── main markdown → PDF renderer ─────────────────────────────────────────────
def render_markdown_to_pdf(md_text, pdf):
    """Walk through markdown lines and render each element."""
    lines = md_text.splitlines()
    i = 0
    N = len(lines)

    while i < N:
        raw = lines[i]
        stripped = raw.strip()

        # ── headings ─────────────────────────────────────────────────────────
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title = sanitize(stripped[2:].strip())
            pdf.ln(4)
            pdf.set_font('Helvetica', 'B', 18)
            pdf.set_text_color(*C_HEADING)
            pdf.multi_cell(pdf.pw, 9, title, align='C',
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.rule()
            i += 1; continue

        if stripped.startswith('### '):
            title = sanitize(stripped[4:].strip())
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(*C_HEADING)
            pdf.multi_cell(pdf.pw, 6, title,
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            i += 1; continue

        if stripped.startswith('## '):
            title = sanitize(stripped[3:].strip())
            pdf.ln(5)
            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_text_color(*C_HEADING)
            pdf.multi_cell(pdf.pw, 7.5, title,
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            # underline
            y = pdf.get_y() + 1
            pdf.set_draw_color(*C_HEADING)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.set_draw_color(*C_RULE)
            pdf.ln(4)
            i += 1; continue

        # ── horizontal rule ---
        if stripped in ('---', '***', '___') or re.match(r'^-{3,}$', stripped):
            pdf.rule()
            i += 1; continue

        # ── image ![alt](path)
        img_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', stripped)
        if img_match:
            img_path = os.path.join(BASE_DIR, img_match.group(2))
            # Next non-blank line starting with ** is the caption
            caption = ''
            j = i + 1
            while j < N and lines[j].strip() == '':
                j += 1
            if j < N and lines[j].strip().startswith('**Figure'):
                cap_raw = lines[j].strip()
                # strip leading ** and trailing *
                cap_raw = re.sub(r'^\*\*', '', cap_raw)
                cap_raw = re.sub(r'\*$', '', cap_raw)
                caption = re.sub(r'\*', '', cap_raw)
                # Remove markdown italic markers
                caption = caption.replace('*', '')
                i = j + 1
            else:
                i += 1
            pdf.insert_figure(img_path, caption.strip())
            continue

        # ── fenced code block ```
        if stripped.startswith('```'):
            code_lines = []
            i += 1
            while i < N and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            pdf.write_code_block(code_lines)
            continue

        # ── markdown table (line contains |)
        if '|' in stripped and stripped.startswith('|'):
            table_lines = []
            while i < N and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            headers, rows, col_widths = _parse_md_table(table_lines)
            if headers:
                pdf.write_table(headers, rows, col_widths)
            continue

        # ── blockquote >
        if stripped.startswith('>'):
            content = sanitize(stripped.lstrip('> ').strip())
            pdf.set_fill_color(*C_CODE_BG)
            pdf.set_x(pdf.l_margin + 5)
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(pdf.pw - 5, 5, content, fill=True,
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
            i += 1; continue

        # ── bullet list  - item or * item
        if re.match(r'^[-*]\s+', stripped):
            bullet_text = re.sub(r'^[-*]\s+', '', stripped)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(*C_DARK)
            pdf.set_x(pdf.l_margin + 4)
            # Draw bullet
            pdf.set_x(pdf.l_margin + 2)
            pdf.cell(2, 5.5, '-')
            pdf.set_x(pdf.l_margin + 6)
            segments = _parse_inline(bullet_text)
            for seg_text, seg_bold, seg_italic in segments:
                s = ('B' if seg_bold else '') + ('I' if seg_italic else '')
                pdf.set_font('Helvetica', s, 10)
                pdf.write(5.5, sanitize(seg_text))
            pdf.ln(5.5 * 0.5)
            i += 1; continue

        # ── numbered list  1. item
        if re.match(r'^\d+\.\s+', stripped):
            num = re.match(r'^(\d+)\.\s+', stripped).group(1)
            content = re.sub(r'^\d+\.\s+', '', stripped)
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(*C_DARK)
            pdf.set_x(pdf.l_margin + 2)
            pdf.cell(5, 5.5, f'{num}.')
            pdf.set_x(pdf.l_margin + 8)
            segments = _parse_inline(content)
            for seg_text, seg_bold, seg_italic in segments:
                s = ('B' if seg_bold else '') + ('I' if seg_italic else '')
                pdf.set_font('Helvetica', s, 10)
                pdf.write(5.5, sanitize(seg_text))
            pdf.ln(5.5 * 0.5)
            i += 1; continue

        # ── blank line
        if stripped == '':
            pdf.ln(3)
            i += 1; continue

        # ── metadata lines (bold key-value at top of doc)
        if stripped.startswith('**') and stripped.endswith('**') and ':' not in stripped:
            # pure bold line — render as bold
            text = sanitize(stripped.strip('*').strip())
            pdf.write_paragraph(text, size=10, bold=True)
            pdf.ln(1)
            i += 1; continue

        # ── figure caption line **Figure N.**
        if stripped.startswith('**Figure'):
            # already consumed as part of image block above; if we're here
            # it was a standalone caption line — render in caption style
            cap = sanitize(re.sub(r'\*', '', stripped))
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(*C_CAPTION)
            pdf.multi_cell(pdf.pw, 5, cap.strip(), align='C',
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)
            i += 1; continue

        # ── everything else: paragraph text
        # Collect continuation lines (non-blank, not a special token)
        para_lines = [stripped]
        i += 1
        while i < N:
            nxt = lines[i].strip()
            if (nxt == '' or nxt.startswith('#') or nxt.startswith('```')
                    or nxt.startswith('---') or nxt.startswith('|')
                    or re.match(r'^[-*]\s+', nxt) or re.match(r'^\d+\.\s+', nxt)
                    or re.match(r'^!\[', nxt) or nxt.startswith('>')):
                break
            para_lines.append(nxt)
            i += 1
        para = ' '.join(para_lines)
        if para.strip():
            pdf.set_font('Helvetica', '', 10)
            pdf.set_text_color(*C_DARK)
            segments = _parse_inline(para)
            for seg_text, seg_bold, seg_italic in segments:
                s = ('B' if seg_bold else '') + ('I' if seg_italic else '')
                pdf.set_font('Helvetica', s, 10)
                pdf.set_text_color(*C_DARK)
                pdf.write(5.5, sanitize(seg_text))
            pdf.ln(5.5 * 0.6)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    with open(MD_FILE, 'r', encoding='utf-8') as f:
        md_text = f.read()

    pdf = ReportPDF()
    pdf.add_page()

    render_markdown_to_pdf(md_text, pdf)

    pdf.output(OUT_FILE)
    print(f'PDF written to: {OUT_FILE}')


if __name__ == '__main__':
    main()
