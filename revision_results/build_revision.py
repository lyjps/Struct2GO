"""Build the Word revision and response from the archived experiment results."""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont
from expert_fusion_report import findings, load as load_expert, overall, performance, weak_hit_findings
from extended_validation_report import external_summary, external_table, matched, read
import regional_confidence_report as regional_report


HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "submission"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = Document(sys.argv[1])
NOTES = (HERE / "Manuscript_revision.md").read_text(encoding="utf-8")
RESULTS = (HERE / "RESULTS.md").read_text(encoding="utf-8")
EXPERT = load_expert()
EXTERNAL = read("external_nrpdb_routing.json")
FULL_GO = {branch: read(f"{branch}_full_deepgoplus.json") for branch in ("bp", "cc", "mf")}
FULL_FRI = read("mf_full_deepfri.json")
EXTENDED = any(FULL_GO.values()) or FULL_FRI or EXTERNAL
REGIONAL = regional_report.load()
STANDARDIZED = regional_report.load("standardized_confidence")
TITLE = ("Testing Structural Confidence Beyond Sequence Similarity in Protein Function Prediction"
         if EXPERT else "Structural Confidence in Sequence and Structure Fusion for Protein Function Prediction")
if REGIONAL:
    TITLE = "Testing Where AlphaFold Confidence Helps Protein Function Prediction"
FOLLOWUP = HERE / "confidence_followup_manuscript.md"


def make_document():
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Inches(1)
    section.top_margin = section.bottom_margin = Inches(0.85)
    for name, size in (("Normal", 11), ("Title", 17), ("Heading 1", 13),
                       ("Heading 2", 11), ("Caption", 10)):
        font = doc.styles[name].font
        font.name = "Times New Roman"
        font.size = Pt(size)
        font.color.rgb = RGBColor(0, 0, 0)
        for border in doc.styles[name]._element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)
    normal = doc.styles["Normal"].paragraph_format
    normal.space_after = Pt(6)
    normal.line_spacing = 1.12
    return doc


def figure():
    canvas = Image.new("RGB", (1800, 720), "white")
    draw = ImageDraw.Draw(canvas)
    font_dir = Path("C:/Windows/Fonts")
    font = ImageFont.truetype(str(font_dir / "arial.ttf"), 32)
    small = ImageFont.truetype(str(font_dir / "arial.ttf"), 29)
    bold = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 37)
    ink = "#172b42"

    def box(x, y, w, h, label, fill):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill,
                               outline="#7c8c9b", width=3)
        draw.multiline_text((x + w / 2, y + h / 2), label, font=font, fill=ink,
                            anchor="mm", align="center", spacing=8)

    def arrow(x, y, end):
        draw.line((x, y, end, y), fill=ink, width=5)
        draw.polygon([(end, y), (end - 18, y - 11), (end - 18, y + 11)], fill=ink)

    draw.text((20, 15), "A   Fusion after encoding", font=bold, fill=ink)
    for y, left, mid, fill in [
        (77, "Protein sequence", "Sequence encoder", "#eaf1fa"),
        (197, "Residue contact graph", "Structure encoder", "#e9f4ed"),
        (317, "Per-residue pLDDT", "Mean pLDDT / 100", "#f9f0dc"),
    ]:
        box(20, y, 350, 86, left, fill)
        box(420, y, 360, 86, mid, fill)
        arrow(370, y + 43, 420)
        arrow(780, y + 43, 870)
    box(870, 77, 535, 326, "Project and normalize embeddings\n\n"
        "Structural weight\na = sigmoid(wc + b)\n\nWeighted sum of embeddings", "#edf1f5")
    arrow(1405, 240, 1470)
    box(1470, 197, 310, 86, "GO term scores", "#edf1f5")
    draw.line((20, 449, 1780, 449), fill="#c6ced6", width=3)
    draw.text((20, 478), "B   Optional prior inside graph attention", font=bold, fill=ink)
    box(20, 546, 490, 105, "Edge i → j\nResidue scores pᵢ and pⱼ", "#f9f0dc")
    arrow(510, 598, 580)
    box(580, 546, 650, 105, "Add log(max((pᵢ + pⱼ)/2, ε))\nto the ordinary edge logit", "#e9f4ed")
    arrow(1230, 598, 1300)
    box(1300, 546, 480, 105, "Normalize across\nincoming edges", "#e9f4ed")
    draw.text((20, 681), "Static control: learn a single structural weight for all proteins.", font=small, fill=ink)
    path = OUT / "Figure_1.png"
    canvas.save(path, dpi=(300, 300))
    return path


def expert_figure():
    image = Image.new("RGB", (1800, 620), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 33)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 38)
    ink = "#172b42"

    def node(bounds, text, fill):
        draw.rounded_rectangle(bounds, radius=13, fill=fill, outline="#7c8c9b", width=3)
        x1, y1, x2, y2 = bounds
        draw.multiline_text(((x1+x2)/2, (y1+y2)/2), text, font=font, fill=ink,
                            anchor="mm", align="center", spacing=8)

    def right(start, end, y):
        draw.line((start, y, end, y), fill=ink, width=4)
        draw.polygon(((end, y), (end-18, y-11), (end-18, y+11)), fill=ink)

    draw.text((26, 20), "Score fusion with alignment evidence", font=bold, fill=ink)
    for i, label in enumerate(("DIAMOND label transfer", "Sequence CNN", "Pooled SeqVec classifier", "Contact graph classifier")):
        y = 85 + i*106
        node((26, y, 465, y+80), label, "#eaf1fa" if i < 3 else "#e9f4ed")
        right(465, 590, y+40)
    node((590, 85, 1120, 483), "Combine expert GO probabilities\n\n"
         "One softmax weight per expert\n\nFit on validation proteins", "#edf1f5")
    node((1240, 99, 1774, 281), "Gate inputs\nAlignment detection and identity\nOptional mean pLDDT", "#f9f0dc")
    draw.line((1120, 190, 1240, 190), fill=ink, width=4)
    draw.polygon(((1120, 190), (1138, 179), (1138, 201)), fill=ink)
    right(1120, 1240, 398)
    node((1240, 348, 1774, 448), "Fused GO term scores", "#edf1f5")
    draw.multiline_text((26, 527), "Static, real pLDDT, and shuffled pLDDT controls reuse\n"
                        "the same frozen expert predictions.", font=font, fill=ink)
    path = OUT / "Figure_2.png"
    image.save(path, dpi=(300, 300))
    return path


def pooling_figure():
    image = Image.new("RGB", (1800, 530), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 33)
    bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 38)
    ink = "#172b42"

    def node(x, y, width, height, label, fill):
        draw.rounded_rectangle((x, y, x + width, y + height), radius=14, fill=fill, outline="#7c8c9b", width=3)
        draw.multiline_text((x + width / 2, y + height / 2), label, font=font, fill=ink,
                            anchor="mm", align="center", spacing=8)

    def arrow(x, y, end):
        draw.line((x, y, end, y), fill=ink, width=4)
        draw.polygon(((end, y), (end-17, y-10), (end-17, y+10)), fill=ink)

    draw.text((26, 16), "Matched tests of residue confidence pooling", font=bold, fill=ink)
    node(26, 162, 408, 192, "Frozen graph encoder\n\nSame residue embeddings", "#e9f4ed")
    node(1326, 162, 448, 192, "Separate matched readouts\n\nSeqVec + ordinary + extra", "#eaf1fa")
    draw.line((434, 258, 495, 258), fill=ink, width=4)
    draw.line((495, 119, 495, 397), fill=ink, width=4)
    draw.line((1261, 119, 1261, 397), fill=ink, width=4)
    arrow(1261, 258, 1326)
    for y, label in ((76, "Ordinary mean copied"), (215, "Real normalized pLDDT weights"),
                     (354, "Weights permuted within protein")):
        arrow(495, y + 43, 568)
        node(568, y, 615, 86, label, "#f9f0dc")
        draw.line((1183, y + 43, 1261, y + 43), fill=ink, width=4)
    draw.text((26, 480), "The permutation preserves each protein’s confidence distribution.", font=font, fill=ink)
    path = OUT / "Figure_3.png"
    image.save(path, dpi=(300, 300))
    return path


def table(doc, rows, widths=None):
    grid = doc.add_table(rows=0, cols=len(rows[0]))
    grid.style = "Table Grid"
    grid.autofit = False
    widths = widths or (3.20, 1.10, 1.10, 1.10)
    for column, width in zip(grid.columns, widths):
        column.width = Inches(width)
    for idx, row in enumerate(rows):
        entry = grid.add_row()
        props = entry._tr.get_or_add_trPr()
        props.append(OxmlElement("w:cantSplit"))
        if idx == 0:
            props.append(OxmlElement("w:tblHeader"))
        for j, (cell, value) in enumerate(zip(entry.cells, row)):
            cell.width = Inches(widths[j])
            if idx and j >= 3 and rows[0][3] == "Real − static":
                value = value.replace(" [", "\n[").replace("-", "−")
            cell.text = value
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(3)
            paragraph.paragraph_format.space_before = Pt(3)
            paragraph.paragraph_format.keep_with_next = idx < len(rows) - 1 and (
                len(rows) < 12 or idx % 6 != 0 or idx == 0
            )
            if j:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(9)
                run.bold = idx == 0
            if idx == 0:
                fill = OxmlElement("w:shd")
                fill.set(qn("w:fill"), "E9EDF2")
                cell._tc.get_or_add_tcPr().append(fill)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), "D9D9D9")
        borders.append(edge)
    grid._tbl.tblPr.append(borders)


def result_table(doc, heading, caption, widths=None):
    block = RESULTS.split("## " + heading + "\n", 1)[1].split("\n## ", 1)[0]
    rows = [[part.strip() for part in line.strip("|").split("|")]
            for line in block.splitlines() if line.startswith("|")]
    label = doc.add_paragraph(caption, "Caption")
    label.paragraph_format.keep_with_next = True
    table(doc, rows[:1] + rows[2:], widths)


REFERENCES = [
    ("Gene Ontology Consortium. The Gene Ontology resource: enriching a GOld mine. Nucleic Acids Res. 49, D325–D334 (2021).", "10.1093/nar/gkaa1113"),
    ("Kulmanov and Hoehndorf. DeepGOPlus: improved protein function prediction from sequence. Bioinformatics 36, 422–429 (2020).", "10.1093/bioinformatics/btz595"),
    ("Gligorijević et al. Structure-based protein function prediction using graph convolutional networks. Nat. Commun. 12, 3168 (2021).", "10.1038/s41467-021-23303-9"),
    ("Jiao et al. Struct2GO: protein function prediction based on graph pooling algorithm and AlphaFold2 structure information. Bioinformatics 39, btad637 (2023).", "10.1093/bioinformatics/btad637"),
    ("Lai and Xu. Accurate protein function prediction via graph attention networks with predicted structure information. Brief. Bioinform. 23, bbab502 (2022).", "10.1093/bib/bbab502"),
    ("Meng and Wang. TAWFN: a deep learning framework for protein function prediction. Bioinformatics 40, btae571 (2024).", "10.1093/bioinformatics/btae571"),
    ("Liu et al. Multistage attention-based extraction and fusion of protein sequence and structural features for protein function prediction. Bioinformatics 41, btaf374 (2025).", "10.1093/bioinformatics/btaf374"),
    ("Lee et al. MEGA-GO: functions prediction of diverse protein sequence length using Multi-scalE Graph Adaptive neural network. Bioinformatics 41, btaf032 (2025).", "10.1093/bioinformatics/btaf032"),
    ("Structure-guided sequence representation learning for generalizable protein function prediction. Bioinformatics 41, btaf511 (2025).", "10.1093/bioinformatics/btaf511"),
    ("Char et al. ProtNote: a multimodal method for protein–function annotation. Bioinformatics 41, btaf170 (2025).", "10.1093/bioinformatics/btaf170"),
    ("A unified multimodal model for generalizable zero-shot and supervised protein function prediction. Bioinformatics 42, btag356 (2026).", "10.1093/bioinformatics/btag356"),
    ("Danner et al. Genolator enables protein function interpretation using a multimodal large language model fusing genomic and structural interpretation with natural language interaction. Genome Biol. 27, 290 (2026).", "10.1186/s13059-026-04274-w"),
    ("Jumper et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021).", "10.1038/s41586-021-03819-2"),
    ("AlphaFold Protein Structure Database in 2024. Nucleic Acids Res. 52, D368–D375 (2024).", "10.1093/nar/gkad1011"),
    ("Heinzinger et al. Modeling aspects of the language of life through transfer-learning protein sequences. BMC Bioinformatics 20, 723 (2019).", "10.1186/s12859-019-3220-8"),
    ("Buchfink, Reuter and Drost. Sensitive protein alignments at tree-of-life scale using DIAMOND. Nat. Methods 18, 366–368 (2021).", "10.1038/s41592-021-01101-x"),
    ("Arevalo et al. Gated multimodal units for information fusion (2017).", "arXiv:1702.01992"),
    ("Jain and Wallace. Attention is not explanation. Proc. NAACL-HLT, 3543–3556 (2019).", "10.18653/v1/N19-1357"),
    ("Fast and accurate protein function prediction from sequence through pretrained language model and homology-based label diffusion. Brief. Bioinform. 24, bbad117 (2023).", "10.1093/bib/bbad117"),
    ("Shi et al. Struct2GO-Enhanced: Multimodal Graph Attention Improves Protein Function Prediction. J. Chem. Inf. Model. 66, 1241–1249 (2026).", "10.1021/acs.jcim.5c02419"),
]


def add_text(doc, value):
    for paragraph in value.strip().split("\n\n"):
        if paragraph.startswith("### "):
            doc.add_heading(paragraph[4:], level=2)
        elif paragraph.startswith("|"):
            rows = [[cell.strip() for cell in line.strip("|").split("|")]
                    for line in paragraph.splitlines()]
            table(doc, rows[:1] + rows[2:], (0.75, 1.35, 0.50, 1.30, 1.30, 1.30))
        elif paragraph.startswith("Table 4."):
            doc.add_paragraph(paragraph, "Caption").paragraph_format.keep_with_next = True
        elif paragraph.startswith("BPO real-confidence"):
            doc.add_paragraph(paragraph).paragraph_format.keep_together = True
        elif paragraph.strip():
            doc.add_paragraph(paragraph.strip())


def archived_paragraphs(start, stop):
    block = NOTES.split(start, 1)[1].split(stop, 1)[0]
    return [re.sub(r" \[[^\]]+\]\(.*$", "", p.strip())
            for p in block.strip().split("\n\n") if p.strip()]


def manuscript():
    doc = make_document()
    doc.add_paragraph(TITLE, "Title")
    for idx in (1, 2):
        doc.add_paragraph(SOURCE.paragraphs[idx].text.strip())
    doc.add_heading("Abstract", 1)
    followup_text = FOLLOWUP.read_text(encoding="utf-8") if FOLLOWUP.exists() else ""
    abstract = ("AlphaFold predicts a structure and estimates its local accuracy at each residue. "
                "We tested whether these estimates improve sequence and structure fusion for Gene Ontology (GO) "
                "prediction. A late gate uses mean predicted local distance difference test (pLDDT) to weight pooled "
                "sequence and structure features. An optional graph attention prior uses the confidence of residues "
                "at each edge. On the Struct2GO-derived partitions, after removing exact cross-partition sequence "
                "overlaps, the late gate increased mean test Fmax by 0.0012–0.0059 relative to the corresponding "
                "static mixtures across two compact encoders and three GO branches. The residue prior changed "
                "mean Fmax by at most 0.00008. An adapted TAWFN experiment found similar performance with constant "
                "and confidence-dependent score fusion. A compact DeepGOPlus reference outscored the tested fusion "
                "variants from the original confidence experiments across all branches.")
    if followup_text:
        abstract += (" In the compact GCN follow-up, real and shuffled pLDDT yielded similar overall F1 in all "
                     "three branches. An exploratory high-confidence CCO subgroup showed a small positive difference. "
                     "The results provide limited evidence for a benefit from protein-specific confidence.")
    if EXPERT:
        effects = {
            key: [run["contrasts"][key]["all"]["mean_delta"] for run in EXPERT.values()]
            for key in ("sequence_routed_minus_deepgoplus_control", "structure_routed_minus_sequence_routed",
                        "confidence_routed_minus_shuffled_routed")
        }
        blend = ", ".join(f"{branch} {delta:+.4f}" for branch, delta in
                          zip(EXPERT, effects["sequence_routed_minus_deepgoplus_control"]))
        graph = max(map(abs, effects["structure_routed_minus_sequence_routed"]))
        confidence = max(map(abs, effects["confidence_routed_minus_shuffled_routed"]))
        abstract = (
            "AlphaFold estimates local structural accuracy for each residue. We investigated whether these estimates "
            "improve Gene Ontology prediction beyond sequence information. On exact-sequence-filtered Struct2GO-derived "
            "partitions, a confidence gate slightly improved test Fmax over static fusion, and the residue attention prior "
            "had negligible effect. A shuffled-confidence control reproduced most of the gate gains. We then designed an "
            "alignment-guided mixture of frozen experts and separated validation proteins for checkpoint selection, mixture "
            "fitting, and threshold selection. Compared with a compact CNN and DIAMOND blend, the sequence mixture changed "
            f"held-out F1 by {blend} and improved micro AUPR. Adding a contact graph expert changed mean F1 by at most "
            f"{graph:.4f} in absolute value across branches. Real versus shuffled pLDDT changed mean F1 by at most "
            f"{confidence:.4f}. The tested sequence mixture improved ranking on this benchmark, with limited incremental "
            "value from the graph expert and structural confidence."
        )
        if EXTERNAL and all(FULL_GO.values()) and FULL_FRI:
            abstract = (
                "We tested whether AlphaFold confidence improves Gene Ontology prediction beyond sequence information. "
                "On exact-sequence-filtered Struct2GO-derived splits, shuffled confidence reproduced most of the gains "
                "of the original confidence gate. We then fitted an alignment-guided mixture of frozen experts with "
                "separate validation subsets for checkpoint selection, fusion fitting, and threshold selection. "
                "Against full-width DeepGOPlus retrained on the same splits, the sequence mixture changed held-out F1 "
                "by −0.0023 for biological process, +0.0065 for cellular component, and +0.0128 for molecular function, "
                "and increased micro AUPR in all three branches. The full molecular-function DeepFRI configuration "
                "reached F1 of 0.6348. Adding a contact graph expert or replacing shuffled pLDDT with real pLDDT changed "
                "mean F1 by at most 0.0002 or 0.0005, respectively. On the independent nrPDB-GO test set, a two-expert "
                "replication found no fixed-threshold F1 gain over a grid-tuned static blend in any branch. For "
                "cellular component, F1 fell by 0.0198 while Fmax increased by 0.0114. These results show that "
                "alignment routing can improve performance within a benchmark, with gains that depend on the "
                "cohort, available experts, and operating threshold."
            )
    if REGIONAL:
        abstract = (
            "AlphaFold supplies a local confidence score, pLDDT, for each residue. We tested three ways of using that "
            "signal for Gene Ontology prediction on exact-sequence-filtered Struct2GO-derived splits. We used mean "
            "pLDDT as a direct structure weight, fitted routing rules from regional summaries, and added a "
            "confidence-weighted view of a frozen contact graph. Shuffling the residue weights within each protein "
            "preserved its confidence distribution while testing the value of assigning confidence to the correct residues. "
            "Relative to a learned static blend, direct weighting changed held-out F1 by "
            + regional_report.deltas(REGIONAL, "direct_mean_minus_static")
            + ". The validation-selected fallback used the sequence expert for all test proteins. "
            "Real residue weights changed F1 relative to shuffled weights by "
            + regional_report.deltas(REGIONAL, "real_pool_minus_shuffled_pool")
            + ". We also compared against full-width DeepGOPlus and the full DeepFRI molecular-function configuration. "
            "Alignment-guided sequence fusion improved fixed-threshold F1 in two branches on the local benchmark. "
            "Its two-expert replication on nrPDB-GO did not improve fixed-threshold F1 over tuned static fusion. "
            "These controls separate gains from score fusion from any benefit of assigning structural confidence to the correct protein or residue."
        )
        if STANDARDIZED:
            abstract += (" After standardizing the readout inputs, the real-versus-shuffled differences were "
                         + regional_report.deltas(STANDARDIZED, "real_pool_minus_shuffled_pool") + ".")
    add_text(doc, abstract)
    doc.add_paragraph("Keywords  Protein function prediction, Gene Ontology, AlphaFold, pLDDT, multimodal fusion")
    doc.add_heading("1. Introduction", 1)
    intro = archived_paragraphs("## Introduction replacement", "## Methods replacement")
    add_text(doc, "The Gene Ontology groups protein annotations into biological process (BPO), cellular component (CCO), and molecular function (MFO) [1]. " + intro[0] + " [2–5]\n\n" + intro[1] + " [6–12]\n\nOur contribution is an evaluation of predicted structural confidence as a conditioning signal. Gated fusion is established [17]. We test a simple gate against a static mixture and measure the incremental effect of a residue prior under each fusion rule.")
    if EXPERT:
        add_text(doc, "Struct2GO-Enhanced refined graph feature extraction and pooling [20]. SPROF-GO combines pretrained sequence representations with homology-based label diffusion [19]. These observations motivate a stronger test of structural information. We ask whether alignment evidence improves the weighting of fixed experts, whether the graph expert contributes after the sequence experts are available, and whether correctly assigned pLDDT improves that mixture over shuffled pLDDT.")
    doc.add_heading("2. Methods", 1)
    add_text(doc, """### 2.1 Dataset and structural provenance

We used the archived Struct2GO-derived sequence features, GO labels, and contact graphs [4]. We decoded sequences from the one-hot residue features. We kept the training partitions intact, removed exact training sequence matches from validation, and removed test matches to either earlier partition. Related sequences may remain across partitions.

The preprocessing documentation identifies AlphaFold DB as the structure source [13,14]. The processed archive contains one contact graph and one pLDDT vector per protein. The database release, model identifiers, and any upstream candidate selection rule could not be recovered. We averaged the stored pLDDT scores across residues and divided by 100 to obtain protein confidence c. We could not estimate variation across alternate predicted models from this archive. Different predictions can alter both contacts and pLDDT, so any future ensemble comparison should pair each graph with its own confidence vector.

### 2.2 Fusion and residue confidence

The revised UAX fusion module independently projects the sequence and structure embeddings into a common space and applies layer normalization. With mean confidence c on the unit interval, the structure weight is a = sigmoid(wc + b). The classifier receives (1 − a) times the projected sequence embedding plus a times the projected structure embedding. The static control supplies zero in place of c and learns one global mixture weight. The same scalar rule can blend predicted probabilities from separate fixed encoders.

The optional prior adds log(max((pᵢ + pⱼ)/2, 10⁻⁶)) to the post-LeakyReLU logit of contact edge i → j before normalization. Here pᵢ and pⱼ are unit-scale residue confidences. This differs from multiplying signed logits by confidence, as in the earlier implementation. The revised attention experiment crosses ordinary versus prior-weighted attention with static versus confidence fusion. It uses the same graph convolution input, residual attention layer, mean pooling, and classifier across all four conditions.

The earlier cross-attention block operated on a single pooled key and value. Softmax over one key returns a weight of one, so that operation reduces to a value projection and provides no query-dependent token selection. We removed the block from the revised model and use the matched static versus confidence comparison to isolate the gate. Neither the gate nor the attention coefficients identify functional residues [18].""")
    image_par = doc.add_paragraph()
    image_par.paragraph_format.keep_with_next = True
    image_par.add_run().add_picture(str(figure()), width=Inches(6.48))
    doc.add_paragraph("Figure 1. The gate consumes protein-level outputs after encoding. The optional prior acts inside graph attention and requires residue-aligned confidence and access to edge logits.", "Caption")
    doc.add_heading("2.3 Baselines and evaluation", 2)
    methods = archived_paragraphs("### Baseline comparison and split control", "## Results replacement")
    add_text(doc, methods[1].replace("Evaluate", "We evaluated", 1).replace("retrain", "we retrained", 1).replace("Pool over positions and take", "The model pools over positions and takes").replace("Use DIAMOND", "We ran DIAMOND").replace("combine training labels", "combined training labels").replace("and choose", "and chose").replace("Report DIAMOND alone and the CNN alone alongside the blend.", "We report the two standalone baselines alongside the blend.").replace("use its graph-convolution", "we used its graph-convolution") + " [2,3,15,16]")
    add_text(doc, "For the recent-backbone comparison, we adapted the released TAWFN convolutional and adaptive graph-convolutional blocks to MFO with narrower layers [6] (source commit 2aaf850). We supplied amino-acid tokens and a projection of pooled SeqVec at each residue in place of native contextual residue embeddings, used the archived contact edges, and reduced graph pooling width. The two predictors were trained separately. We fitted each score blend on validation probabilities with the encoders fixed.")
    add_text(doc, """The matched GCN uses a graph convolution input layer and a residual graph convolution, followed by mean pooling. The matched GAT substitutes graph attention for the second convolution. We trained the matched fusion variants with AdamW, a learning rate of 0.001, weight decay of 0.0001, width 48, batches of 96 proteins, and three paired seeds. We allowed up to eight epochs and selected the checkpoint by validation binary cross-entropy.

Protein-centric Fmax is the maximum over thresholds from 0.01 to 1.00, in 0.01 steps. Precision is averaged across proteins with predictions and recall across proteins with labels. We also report F1 at the validation-selected threshold and trapezoidal micro AUPR in the accompanying results. We did not propagate scores to GO ancestors. The original tables were superseded because the archived evaluation code and metric definitions differ from this protocol.""")
    if EXPERT:
        doc.add_heading("2.4 Alignment guided expert fusion", 2)
        add_text(doc, """Sequence similarity is an established source of evidence for GO annotation [2,5,19]. To test the incremental value of a graph expert and pLDDT under a stronger sequence comparator, we trained a sequence CNN, a pooled SeqVec classifier, and a contact graph classifier separately. A fourth expert transferred training annotations through DIAMOND bit-score weighting. We fitted a score mixture with weights softmax(Wx + b) shared across GO terms for each protein. The feature vector x contained a detectable-hit indicator and the largest sequence identity among hits covering at least 70% of both proteins. When no hit met that coverage criterion, the identity feature was set to zero. Confidence conditions added mean pLDDT or independently permuted pLDDT. A constant confidence input is absorbed by the intercept in the alignment-only gate, and the shuffled control preserves the feature count and marginal distribution.

We split the original validation partition into disjoint stopping, mixture fitting, and threshold selection subsets of approximately 50%, 30%, and 20%, using a fixed random partition. We selected expert checkpoints on the first subset, froze their outputs, and fitted the mixture on the second subset by minimizing binary cross-entropy with a 0.001 mean-squared penalty on slopes. The same mixture fitting subset selected the compact CNN and DIAMOND blend weight. We selected each F1 operating threshold on the third subset. We trained each neural expert with three paired seeds, eight epochs at most, and a maximum of 64 proteins per GCN or SeqVec minibatch. DIAMOND queried only the training set for annotation transfer and alignment features.

We defined a close hit as at least 40% identity over a hit spanning at least 70% of the query and subject. The test subgroup lacking a close hit can still contain remote relatives or alignments that cover only part of the proteins. This subgroup analysis is a similarity screen within the existing benchmark. We designed this extension after studying the earlier results on the same data.""")
        visual = doc.add_paragraph()
        visual.paragraph_format.keep_with_next = True
        visual.add_run().add_picture(str(expert_figure()), width=Inches(6.48))
        doc.add_paragraph("Figure 2. The alignment guided mixture combines frozen expert scores and tests the contribution of pLDDT after accounting for alignment evidence.", "Caption")
    if EXTENDED:
        doc.add_heading("2.5 Full configurations and independent replication", 2)
        if any(FULL_GO.values()):
            add_text(doc, "We retrained the full-width DeepGOPlus CNN on the same exact-sequence-filtered splits [2]. The PyTorch port uses the published 21-channel encoding, 2,000-residue input window, 16 linear convolution widths from 8 to 128, 512 filters at each width, global max pooling, and a sigmoid multilabel head. We used Adam with a learning rate of 0.0003, a maximum of 12 epochs, length-grouped batches of up to 32, and validation-loss checkpoint selection. Training truncates long sequences and inference combines overlapping windows by taking the maximum score per term. We fitted the DIAMOND mixture on validation data. These results use the published architecture and the local splits, with the implementation and batching changes specified here.")
        if FULL_FRI:
            add_text(doc, "We retrained the full molecular-function DeepFRI configuration on the matched local graphs [3]. The model uses the official frozen two-layer LSTM, a 1,024-channel input projection, three 512-channel graph convolutions with ReLU as in the released molecular-function checkpoint, concatenated sum pooling, a 1,024-unit hidden layer, and the two-logit output for each GO term. With the published weights loaded, the PyTorch port reproduced all 942 probabilities of the released reference example within 4 × 10⁻⁸ absolute error. We then initialized a new graph classifier for the local output labels and used the stored predicted contact graphs. The frozen residue embeddings were cached in float16, and graph batches were grouped by sequence length. We used Adam at 0.0002 with β₁ = 0.95 and β₂ = 0.99, graph-layer L2 regularization of 0.00002, dropout of 0.3, a 50-epoch cap, and early stopping with patience five.")
        if EXTERNAL:
            audit = EXTERNAL["audit"]
            add_text(doc, f"We replicated the alignment routing comparison on the published DeepFRI nrPDB-GO split [3]. "
                     f"We trained on {audit['train']['kept']:,} chains and used {audit['valid']['kept']:,} chains for the three validation roles. "
                     f"The test set retained {audit['test']['kept']:,} of {audit['test']['total']:,} released chains after excluding exact sequence matches to our original benchmark. "
                     "We trained a compact joint CNN and computed DIAMOND transfer from the external training chains. We fitted the same alignment gate to the two frozen experts and compared it with grid-tuned static fusion, binary-cross-entropy-fitted static fusion, and a gate fitted with alignment features permuted jointly within partitions. The external test evaluates the routing rule with different sequences and a two-expert input. It uses neither the contact graph expert nor pLDDT.")
    if REGIONAL:
        doc.add_heading("2.6 Direct weighting regional routing and residue pooling", 2)
        add_text(doc, """We trained separate pooled SeqVec and contact graph experts with three paired seeds. We divided validation data into checkpoint selection, fusion calibration, and operating-threshold selection subsets in proportions of about 50%, 30%, and 20%. We froze the experts before calibrating any fusion rules.

The direct rule takes the normalized mean residue pLDDT as the weight on the graph probabilities. We compared it with a static mixture and a learned gate using the same mean. The regional gate uses four inputs per protein: mean pLDDT, fraction of residues with pLDDT at least 90, fraction below 70, and the highest mean confidence across contiguous 32-residue windows, shortened for smaller proteins. The gate is a calibrated softmax mixture. Its shuffled control permutes the four-feature vector jointly among proteins within each partition. The hard fallback chooses the graph expert only if high-confidence coverage and best-window confidence clear validation-selected thresholds. The calibration grid includes always using the sequence expert.

For pooling, we extracted the final residue embeddings from the same frozen graph encoder. The weighted vector is the sum of the residue embeddings times their normalized pLDDT weights, where the weights sum to one. Each readout receives SeqVec, the ordinary mean graph embedding, and a third vector. This third vector is a copy of the ordinary mean, the real confidence-weighted mean, or a mean weighted by a fixed within-protein permutation of the same confidence weights. All three conditions use the same readout width, training budget, and paired seeds. The test asks whether the placement of confidence across residues improves readout from a frozen encoder. It does not test confidence-dependent message passing.""")
        visual = doc.add_paragraph()
        visual.paragraph_format.keep_with_next = True
        visual.add_run().add_picture(str(pooling_figure()), width=Inches(6.48))
        doc.add_paragraph("Figure 3. The ordinary, real confidence, and shuffled confidence readouts share the encoder and differ in their third input.", "Caption")
    doc.add_heading("3. Results", 1)
    result_table(doc, "Filtered split sizes", "Table 1. Split sizes after exact sequence filtering.", (0.8, 1.0, 1.0, 1.0, 1.35, 1.35))
    result_table(doc, "Baselines and external GCN gate, test Fmax", "Table 2. Test Fmax for baseline and GCN fusion comparisons. Values with ± report mean and sample standard deviation across three seeds. Other values are single runs or deterministic alignment transfer.")
    result_table(doc, "Matched attention ablation, test Fmax", "Table 3. Matched attention and fusion ablation. Mean and sample standard deviation across three seeds.")
    add_text(doc, """The late gate changed mean Fmax by +0.00235, +0.00168, and +0.00418 for BPO, CCO, and MFO with GCN, and by +0.00154, +0.00119, and +0.00586 with ordinary GAT. The confidence prior changed mean Fmax by −0.00008 to +0.00004 across the six matched combinations. The GAT gate increased CCO Fmax by 0.00119, while mean F1 at the validation-selected threshold declined from 0.6934 to 0.6924. All tested late-fusion configurations scored below compact DeepGOPlus on every branch.

On MFO with adapted TAWFN, the sequence and structure branches reached Fmax of 0.6705 and 0.6166. Static and confidence score blends reached 0.6720 and 0.6721. At the validation-selected threshold, their F1 scores were 0.6711 and 0.6709. The fitted pLDDT slope was −0.7585, so the gate gave the structural branch less weight at higher pLDDT in this fit. This single-seed comparison establishes attachment to the adapted outputs and provides little evidence of a performance gain.""")
    if followup_text:
        doc.add_heading("3.1 When confidence helps", 2)
        add_text(doc, followup_text)
    if EXPERT:
        doc.add_heading("3.2 Incremental value beyond sequence similarity", 2)
        label = doc.add_paragraph("Table 5. Mean test F1 and micro AUPR across three seeds per branch. Each cell shows F1 followed by micro AUPR. F1 thresholds were selected on the separate operating point partition.", "Caption")
        label.paragraph_format.keep_with_next = True
        f1_rows = performance(EXPERT)
        aupr_rows = performance(EXPERT, "test_micro_aupr")
        table(doc, [f1_rows[0]] + [[f1[0], *(f"{a} / {b}" for a, b in zip(f1[1:], aupr[1:]))]
                                for f1, aupr in zip(f1_rows[1:], aupr_rows[1:])],
              (2.48, 1.34, 1.34, 1.34))
        label = doc.add_paragraph("Table 6. Paired changes in fixed-threshold test F1 and descriptive 95% test-protein bootstrap intervals.", "Caption")
        label.paragraph_format.keep_with_next = True
        table(doc, overall(EXPERT), (2.48, 1.34, 1.34, 1.34))
        add_text(doc, findings(EXPERT))
        add_text(doc, "Among test proteins without a close training hit, " + weak_hit_findings(EXPERT))
    if EXTENDED:
        doc.add_heading("3.3 Full baselines and external validation", 2)
        label = doc.add_paragraph("Table 7. Matched local test F1, Fmax, and micro AUPR. Each cell lists the three metrics in that order. Full baselines are single runs. The alignment-guided sequence mixture is the mean of three seeds.", "Caption")
        label.paragraph_format.keep_with_next = True
        table(doc, matched(EXPERT), (2.18, 1.44, 1.44, 1.44))
        if all(FULL_GO.values()) and FULL_FRI:
            add_text(doc, "Against full-width DeepGOPlus, the alignment-guided mixture changed fixed-threshold F1 by −0.0023 for BPO, +0.0065 for CCO, and +0.0128 for MFO. Micro AUPR differences were +0.0526, +0.0346, and +0.0248. These are descriptive comparisons of a single full-width run with a three-seed mean. The matched DeepFRI run selected epoch 25 and stopped after epoch 30.")
        if EXTERNAL:
            label = doc.add_paragraph("Table 8. Independent nrPDB-GO sequence routing. Each cell lists mean fixed-threshold test F1, Fmax, and micro AUPR, in that order.", "Caption")
            label.paragraph_format.keep_with_next = True
            table(doc, external_table(EXTERNAL), (2.18, 1.44, 1.44, 1.44))
            add_text(doc, external_summary(EXTERNAL))
            add_text(doc, "The largest difference between fixed-threshold and threshold-swept conclusions occurred in CCO. The alignment gate reduced fixed-threshold F1 from 0.4150 to 0.3951, while test Fmax increased from 0.4660 to 0.4774. The higher test Fmax indicates that a different threshold could recover performance on this test set. It does not demonstrate an improvement at the operating point chosen before test evaluation.")
    if REGIONAL:
        doc.add_heading("3.4 Tests of direct regional and residue confidence", 2)
        label = doc.add_paragraph("Table 9. Confidence models and matched controls. Cells show mean test F1 at the validation-selected threshold, test Fmax, and micro AUPR.", "Caption")
        label.paragraph_format.keep_with_next = True
        table(doc, regional_report.performance(REGIONAL), (2.18, 1.44, 1.44, 1.44))
        label = doc.add_paragraph("Table 10. Paired changes in fixed-threshold F1 and descriptive 95% protein bootstrap intervals conditional on the trained models.", "Caption")
        label.paragraph_format.keep_with_next = True
        table(doc, regional_report.contrast_table(REGIONAL), (2.48, 1.34, 1.34, 1.34))
        add_text(doc, regional_report.findings(REGIONAL))
        if len(REGIONAL) == 3:
            add_text(doc, "The direct rule also reduced F1 in every mean-pLDDT stratum. For proteins with mean pLDDT at least 90, the changes relative to the learned static blend were −0.0847 for BPO, −0.0703 for CCO, and −0.1482 for MFO. The full confidence-stratum comparisons are included in the supplement.")
        if STANDARDIZED:
            label = doc.add_paragraph("Table 11. Pooling sensitivity after standardizing inputs. Paired fixed-threshold F1 differences and 95% protein bootstrap intervals.", "Caption")
            label.paragraph_format.keep_with_next = True
            table(doc, regional_report.contrast_table(STANDARDIZED), (2.48, 1.34, 1.34, 1.34))
            add_text(doc, regional_report.standardized_summary(STANDARDIZED))
            if "MFO" in STANDARDIZED:
                add_text(doc, "In MFO, standardization raised ordinary-pooling F1 from 0.4770 to 0.7235. The real-versus-shuffled difference fell from +0.0017 to +0.0003, and its standardized-readout interval included zero. This shows that the initial MFO result is sensitive to the readout setup.")
            if "CCO" in STANDARDIZED:
                add_text(doc, "CCO retained a small real-versus-shuffled difference after standardization (+0.0011). Its improvement over ordinary pooling was +0.0007, with a descriptive interval that included zero. These comparisons provide limited evidence for a contribution from residue placement in CCO and leave the benefit over ordinary pooling uncertain.")
    doc.add_heading("4. Discussion", 1)
    add_text(doc, """The larger gate deltas on MFO motivate a closer look at the branch and the reliability of the two predictors. They do not establish that particular residues explain the branch difference. pLDDT estimates local structural accuracy, while a GO label can depend on domains, localization, interaction partners, or flexible regions. The weak structural branch and strong homology baseline can also limit the value of a scalar gate. The residue prior changes relative incoming edge weights, so similar confidence values among neighbors leave their normalized weights close to ordinary attention. That is a possible mechanism for a small effect, and the ablation measures its net predictive contribution.

The initial confidence experiments use compact adaptations, short training budgets, and within-dataset splits. The late gate requires aligned sequence and structure outputs. Generalization of the confidence pathway to other encoder interfaces, remote homologs, and independent test datasets remains open. The archive supports an analysis of variation across proteins, while repeated structure predictions would be needed to assess variation across AlphaFold candidates.

On the evaluated splits, confidence-dependent fusion made small changes relative to a static gate. The optional residue prior showed negligible incremental value. We therefore retain the prior as an ablation and describe the external gate as a simple, testable way to condition fusion on predicted structural confidence.""")
    if followup_text:
        add_text(doc, "The shuffled control indicates that the overall gains over static fusion do not establish a benefit from protein-specific pLDDT. The high-confidence CCO subgroup showed a small advantage over shuffled confidence in an unadjusted comparison. That exploratory result needs confirmation on an independent split before it can guide model selection.")
    if EXPERT:
        add_text(doc, "The redesigned mixture provides a separate test of whether a structure model or pLDDT adds information after a sequence similarity baseline and two sequence predictors are available. Its results apply to the graph encoder and benchmark used here. A stronger geometric encoder, a separate temporal cohort, and controlled comparisons across several predicted structures would test the scope of the result.")
        if EXTERNAL:
            add_text(doc, "The external nrPDB-GO experiment did not reproduce the local fixed-threshold gains over the grid-tuned blend. Because the replication uses two experts, the results cannot separate the effects of cohort changes from the loss of the SeqVec expert. The CCO Fmax and F1 results also show why threshold-swept scores alone are insufficient for assessing transfer of a selected operating point. The graph and pLDDT pathways still require external validation. Proteins with related sequences can still cross benchmark boundaries.")
        if REGIONAL:
            add_text(doc, "Direct weighting assumes that local structural confidence estimates the relative predictive accuracy of the graph expert. The graph expert had lower F1 than the sequence expert in all branches, and the validation-selected hard fallback used only the sequence expert at test time. The regional gate produced the same fixed-threshold F1 as its shuffled control. The within-protein shuffle tests whether confidence contributes information through its residue placement after graph encoding. It preserves the weight distribution and disrupts spatial clustering of confidence. The conclusions are limited to the linear weighting rule, contiguous windows, and fixed encoder tested here. A regional confidence pattern can reflect disorder or domain composition, so a performance change alone would not establish a biological mechanism.")
            if STANDARDIZED:
                add_text(doc, "The tested pooling design preserves a route for confidence to affect the graph representation without forcing it to set the relative trust in sequence and structure. Its small, readout-dependent effects justify reporting the permutation controls and input scaling together. These experiments do not establish that pLDDT-guided pooling improves the strongest sequence baseline.")
        doc.add_heading("Data and Code Availability", 1)
        add_text(doc, "The accompanying reproducibility files contain the evaluation scripts, command lines, and seed-level results. Reproduction requires the processed Struct2GO-derived graphs, sequence features, annotations, and confidence tensors described in Section 2.1. The archived graphs do not retain all AlphaFold model provenance.")
    doc.add_heading("Acknowledgments", 1)
    add_text(doc, "Ayaan Jain thanks Dr. Chua Chin Heng Matthew and Dr. Khanh Lee for their mentorship. The authors acknowledge OpenAI ChatGPT and Codex assistance with code debugging and manuscript revision.")
    doc.add_heading("References", 1)
    for idx, (ref, doi) in enumerate(REFERENCES, 1):
        entry = doc.add_paragraph(f"[{idx}] {ref} {doi}")
        entry.paragraph_format.space_after = Pt(3)
        entry.paragraph_format.keep_together = True
        for run in entry.runs:
            run.font.size = Pt(10.5)
    return doc


def response():
    doc = make_document()
    doc.add_paragraph("Response to the Reviewers", "Title")
    add_text(doc, """Dear Editor and Reviewers,

Thank you for identifying issues in the scope, evaluation, and presentation of the original submission. We revised the manuscript around a late confidence gate and a separate residue attention ablation. We replaced the original performance claims with results from a consistent evaluation and report where the components offered little benefit. The marked manuscript highlights replacement text and tables.""")
    add_text(doc, """During revision, we found that an archived evaluator instantiated a structure-only GAT control and that the legacy AUPR helper initialized its area at 0.06. Because we could not verify that all original values came from a consistent evaluation, we withdrew those tables and used the independently recomputed metrics in Sections 2.3 and 3.""")
    doc.add_heading("Reviewer 1", 1)
    items = [
        ("Broader comparisons and recent backbones", "Sections 2.3 and 3 add sequence, structure, DIAMOND, compact DeepGOPlus, and compact one-hot DeepFRI references. We also compared static and confidence score blending on adapted TAWFN blocks. We specify the substitutions in inputs and width. Full published configurations and additional independent benchmarks remain outside this evaluation."),
        ("Portability", "Section 2.2 and Figure 1 separate the late gate from the optional attention prior. The late gate consumes embeddings or probabilities, so it does not require access to graph message passing. We limit the tested scope to the reported GCN, GAT, and adapted TAWFN configurations."),
        ("Literature", "Section 1 now situates the work alongside TAWFN, MAEF-GO, MEGA-GO, domain-aware structure methods, ProtNote, FunBind, and Genolator."),
        ("AlphaFold model selection and variance", "Section 2.1 identifies AlphaFold DB as the documented source and explains how pLDDT was processed. The archive preserves one graph per protein and does not record the original database release, model ID, or candidate selection rule. It cannot support a between-model variance estimate. Section 4 discusses the consequences for interpretation."),
        ("Ablation and mixed results", "Table 3 reports the four matched attention and fusion conditions together. The mean effect of the residue prior was near zero under both static and confidence fusion. We discuss the small branch-specific gate changes and the difference between Fmax and validation-threshold performance."),
    ]
    for heading, value in items:
        doc.add_heading(heading, 2)
        add_text(doc, value)
    doc.add_heading("Reviewer 5", 1)
    for heading, value in [
        ("Novelty", "We acknowledge prior work on gating and limit our contribution to evaluating structural confidence as a fusion input and separating it from a residue prior. The abstract and discussion now state the small or negligible gains."),
        ("Gating and cross-attention", "Section 2.2 explains that a single pooled key yields a unit softmax weight. We removed this block from the revised architecture and withdrew the claim of fine-grained cross-modal interaction. The matched static versus confidence ablation isolates the gate in the revised design."),
        ("Interpretability and mechanism", "We withdrew the claim that attention establishes residue importance and distinguish structural confidence from functional relevance. Section 4 discusses the weak structural branch and the possible effects of edge normalization. The manuscript treats these mechanisms as hypotheses rather than biological explanations."),
        ("Figure 1", "We redrew the figure to show both encoder outputs, the protein confidence calculation, the fusion weight, and the location of the optional edge prior."),
        ("Inconsistent residue attention", "We report the limited effect directly in Table 3 and retain this component as an optional ablation. We removed the claim that the original branch pattern demonstrates a consistent or significant benefit."),
    ]:
        doc.add_heading(heading, 2)
        add_text(doc, value)
    if FOLLOWUP.exists():
        doc.add_heading("Additional confidence analysis", 2)
        add_text(doc, "Section 3.1 adds held-out comparisons against shuffled confidence and reports the paired effect by confidence range. The analysis distinguishes the predictive value of confidence from the effect of perturbing an input to the fitted gate.")
        add_text(doc, "The comparison with shuffled confidence showed no clear overall improvement from assigning the correct pLDDT to each protein. We report the isolated high-confidence CCO result as exploratory and reflect these findings in the abstract and discussion.")
    if EXPERT:
        doc.add_heading("Redesigned score fusion and sequence similarity control", 2)
        add_text(doc, "Sections 2.4 and 3.2, Figure 2, and Tables 5 and 6 test a substantially redesigned mixture against matched sequence controls. We use separate validation subsets for checkpoint selection, mixture fitting, and threshold selection, and report a subgroup lacking close training alignments. The principal contrasts isolate the graph expert and then pLDDT.")
        add_text(doc, findings(EXPERT))
    if EXTENDED:
        doc.add_heading("Full baseline configurations and independent benchmark", 2)
        add_text(doc, "Sections 2.5 and 3.3 distinguish full-architecture retraining on the matched local data from the earlier compact controls. We retain the adapted label for the TAWFN comparison because the native processed input graphs were unavailable. The original test driver also fits fusion weights on test labels, which we excluded from our held-out evaluation protocol.")
        if EXTERNAL:
            add_text(doc, "The external nrPDB-GO experiment retrains the sequence experts on the released training split and removes exact sequence matches to our original benchmark from the external test. We report its scope as sequence routing and include both static and shuffled-alignment controls.")
            add_text(doc, "The external experiment did not reproduce a fixed-threshold F1 gain over the grid-tuned blend. CCO declined by 0.0198 F1 even though test Fmax increased. The revised abstract and discussion state this distinction and limit the scope of the portability claim.")
    if REGIONAL:
        doc.add_heading("Direct weighting and residue placement controls", 2)
        add_text(doc, "Sections 2.6 and 3.4 test the direct structural weight, validation-selected fallback, and confidence-weighted pooling. The pooling comparison shuffles confidence within each protein while keeping its graph and encoder fixed. The paired regional control shuffles the joint confidence summaries among proteins. These tests isolate whether confidence placement changes predictive performance.")
        add_text(doc, regional_report.findings(REGIONAL))
    add_text(doc, "Sincerely,\n\nThe authors")
    return doc


doc = manuscript()
doc.save(OUT / "UAX_Revised_Clean.docx")
# Major replacement sections are highlighted in the marked copy.
for i, paragraph in enumerate(doc.paragraphs):
    if i not in (1, 2):
        for run in paragraph.runs:
            if run.text.strip():
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
for grid in doc.tables:
    for row in grid.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
doc.save(OUT / "UAX_Revised_Marked.docx")
response().save(OUT / "UAX_Response_to_Reviewers.docx")
