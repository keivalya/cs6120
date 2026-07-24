#!/usr/bin/env python3
"""report/make_slides.py — Presentation Slide Deck Generator and PDF Compiler.

Generates a 12-slide presentation markdown file (report/slides.md) and compiles
it into a high-quality presentation PDF (report/slides.pdf) using ReportLab.

Outputs:
  - report/slides.md
  - report/slides.pdf
"""
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

REPO_ROOT = Path(__file__).resolve().parents[1]

SLIDE_MARKDOWN_CONTENT = """# Lost in Instruction: Causal Sensitivity and Paraphrase Fragility in Vision-Language-Action Models

**Project Defense & Presentation Slide Deck**  
*Evaluating SmolVLA-500M, OpenVLA-7B, and OpenVLA-OFT-7.5B on LIBERO*

---

## 1. Motivation: Do VLAs Truly Read Instructions?

- **Current VLA Benchmarking Deficit**: Current benchmarks test policies under static canonical prompts. High success rate does not prove language understanding.
- **Visual Shortcutting Risk**: Models can memorize visual scene layouts and execute motor skills while ignoring instruction text.
- **Core Research Goal**: Isolate the causal contribution of language by holding the physical simulator scene strictly fixed ($S_0 = \\text{const}$) while systematically manipulating text prompts.

---

## 2. Benchmark & Diagnostic Probe Taxonomy

- **Scene-Fixed Causal Constraint**: $S_0(I_{\\text{orig}}) \\equiv S_0(I_{\\text{pert}})$, matching physical state hashes across rollouts.
- **Structural & Semantic Probes**:
  - `blank` / `nonsense` $\\to$ Measure baseline causal reliance (CSS metric).
  - `wrong_object` / `wrong_action` / `wrong_task` $\\to$ Measure Object Reliance & Verb Blindness.
- **Paraphrase Robustness (PRIDE Benchmark)**:
  - `para_object`, `para_action`, `para_compositional` $\\to$ Measure semantic ($S_K$) and structural ($S_T$) resilience.

---

## 3. RQ1: Structural & Semantic Causal Reliance

- **Universal Object Sensitivity**: Substituting the target object noun (`wrong_object`) drops Task Success Rate (TSR) to **7.1% across all models**.
- **Verb Ignorance Bias**: Swapping the action verb (`wrong_action`) results in high performance retention: SmolVLA retains **57.5%**, OpenVLA retains **50.0%**, and OpenVLA-OFT retains **97.5%**.
- **Visual Shortcutting in OFT**: OpenVLA-OFT retains **12.5% success on blank instructions** due to FiLM conditioning layers.

---

## 4. RQ2: Paraphrase Fragility & PRIDE Metrics

- **Catastrophic SmolVLA Degradation**: SmolVLA-500M falls from $72.5\\%$ baseline TSR down to **$4.69\\%$ overall paraphrase TSR** (PRIDE score **2.7**).
- **FiLM Conditioning Resilience**: OpenVLA-OFT-7.5B maintains **$74.22\\%$ overall paraphrase TSR** (PRIDE score **65.8**).
- **Compositional Bottleneck**: Compositional paraphrases (`para_compositional`) represent the largest bottleneck across all architectures.

---

## 5. Failure Loci: Planning vs. Execution Breakdown

- **Planning Failure Dominance**: Over $56\\%$ of rollout failures under text perturbations are **Planning Failures** (gripper fails to approach/touch correct target object).
- **SmolVLA**: $56.3\\%$ Planning Failures ($897 / 1594$).
- **OpenVLA**: $58.6\\%$ Planning Failures ($147 / 251$).
- **OpenVLA-OFT**: $58.6\\%$ Planning Failures ($68 / 116$).

---

## 6. RQ3: Kinematic Trajectory Divergence Profile

- **Immediate Failure Divergence**: Destructive prompts (`blank`, `nonsense`, `wrong_object`) cause **kinematic trajectory divergence at $t_{\\div} \\le 2$ steps** ($e(t) > 0.05\\text{m}$).
- **Verb Perturbation Tracking**: `wrong_action` rollouts in OpenVLA-OFT follow canonical paths for $t_{\\div} \\approx 11.3$ steps with minimal final EEF error ($e(T) = 0.020\\text{m}$).

---

## 7. RQ4: Mechanistic Attention Allocation Ratio (AAR)

- **Early Layer Parity**: At Layer 1, Noun and Verb cross-attention are near parity ($\\text{AAR} = 1.12$).
- **Exponential Verb Attention Decay**: Cross-attention to verb tokens decays rapidly across intermediate transformer layers ($l \\ge 8$).
- **Extreme Late-Layer Noun Dominance**: Achieves a **$12.14\\times$ Attention Allocation Ratio (AAR)** favoring object nouns in late action-generation layers, explaining verb-blindness mechanistically.

---

## 8. RQ5: Inference-Time Contrastive Guidance (ICAG)

- **Training-Free Logit Modulation**:
  $$L_{\\text{guided}}(a_t \\mid O_t, I) = L(a_t \\mid O_t, I) + \\alpha \\cdot \\Big( L(a_t \\mid O_t, I) - L(a_t \\mid O_t, I_{\\text{blank}}) \\Big)$$
- **Performance Recovery at $\\alpha = 0.5$**:
  - SmolVLA: Paraphrase TSR jumps from **$4.69\\% \\to 28.50\\%$** (PRIDE score $2.70 \\to 26.80$).
  - OpenVLA: Paraphrase TSR jumps from **$44.22\\% \\to 68.00\\%$** (PRIDE score $33.30 \\to 59.20$).
  - OpenVLA-OFT: Paraphrase TSR jumps from **$74.22\\% \\to 88.50\\%$** (PRIDE score $65.80 \\to 82.50$).

---

## 9. RQ6: Long-Horizon Multi-Step Task Generalization

- **LIBERO-10 Sequential Dependency**: Multi-step tasks amplify initial language planning errors.
- **SmolVLA**: Compositional paraphrase TSR drops from $1.78\\%$ (Goal) to $0.50\\%$ (LIBERO-10).
- **OpenVLA**: Compositional paraphrase TSR drops from $24.00\\%$ (Goal) to $10.00\\%$ (LIBERO-10).
- **OpenVLA-OFT**: Compositional paraphrase TSR drops from $54.00\\%$ (Goal) to $32.00\\%$ (LIBERO-10).

---

## 10. Summary of Architectural Recommendations

1. **Incorporate Asymmetric Contrastive Objectives**: Train action heads with contrastive loss against unconditioned visual features to eliminate visual shortcutting.
2. **Layer-Wise Verb Attention Boosting**: Add explicit verb-token cross-attention loss penalties to prevent verb attention decay across transformer depth.
3. **Deploy Inference-Time ICAG**: Use ICAG logit modulation ($\\alpha = 0.5$) at deployment for instant paraphrase resilience without re-training.

---

## 11. Conclusion & Key Takeaways

- **Causal Isolation is Essential**: Scene-fixed causal evaluations expose visual shortcutting hidden by standard static benchmarks.
- **Mechanistic Root Cause Identified**: Verb blindness stems from late-layer cross-attention decay ($\\text{AAR} = 12.14\\times$).
- **Simple, Training-Free Remedy**: ICAG logit modulation effectively rescues paraphrase degradation across scales.
"""

def generate_pdf_presentation(slides_text: str, pdf_path: Path):
    """Render slides markdown into landscape PDF presentation slides."""
    styles = getSampleStyleSheet()
    
    slide_title_style = ParagraphStyle(
        'SlideTitle', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=18, leading=22,
        textColor=colors.HexColor('#1a2530'), spaceAfter=14
    )
    slide_body_style = ParagraphStyle(
        'SlideBody', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=16,
        textColor=colors.HexColor('#2c3e50'), spaceAfter=8
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(letter),
        leftMargin=36, rightMargin=36,
        topMargin=36, bottomMargin=36
    )

    story = []
    slides = slides_text.split("---")

    for slide in slides:
        lines = slide.strip().splitlines()
        if not lines:
            continue

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith("# "):
                title_text = line_str[2:].replace("**", "").replace("*", "")
                story.append(Paragraph(f"<b>{title_text}</b>", slide_title_style))
                story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#3498db'), spaceAfter=12))
            elif line_str.startswith("## "):
                stitle_text = line_str[3:].replace("**", "").replace("*", "")
                story.append(Paragraph(f"<b>{stitle_text}</b>", slide_title_style))
                story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#34495e'), spaceAfter=10))
            else:
                p_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line_str)
                p_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', p_text)
                p_text = p_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                p_text = p_text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
                p_text = p_text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
                story.append(Paragraph(p_text, slide_body_style))

        story.append(PageBreak())

    doc.build(story)
    print(f"wrote {pdf_path}")

def main():
    slides_md = REPO_ROOT / "report" / "slides.md"
    slides_pdf = REPO_ROOT / "report" / "slides.pdf"

    slides_md.write_text(SLIDE_MARKDOWN_CONTENT)
    print(f"wrote {slides_md}")

    generate_pdf_presentation(SLIDE_MARKDOWN_CONTENT, slides_pdf)

if __name__ == "__main__":
    main()
