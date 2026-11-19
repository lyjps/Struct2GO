"""Build the confidence study from the completed UAX experiment records."""

import json
import sys
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


RESULTS = Path(__file__).resolve().parent
BRANCHES = {"bp": "BPO", "cc": "CCO", "mf": "MFO"}
ARCHIVED_FILES = {
    b: f"{b}_uax_residue_archived_test{'_corrected' if b == 'bp' else ''}.json"
    for b in BRANCHES
}


def record(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def by_arm(run):
    return {row["variant"]: row for row in run["metrics"]}


def metric_row(label, run):
    models = by_arm(run)
    scores = [f"{models[arm]['test_fmax']:.4f}" for arm in ("sequence", "neutral", "aligned", "shuffled")]
    contrast = run["aligned_minus_control"]["neutral"]["all"]
    lo, hi = contrast["test_protein_bootstrap_95"]
    effect = f"{100 * contrast['mean_delta']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"
    return [label, str(run["counts"]["test"]["included"]), *scores, effect]


def format_doc(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = section.right_margin = Inches(0.90)
    section.top_margin = Inches(0.73)
    section.bottom_margin = Inches(0.73)
    section.footer_distance = Inches(0.38)
    for name, size in (("Normal", 11), ("Title", 22), ("Heading 1", 13),
                       ("Heading 2", 11), ("Caption", 9.5)):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.italic = False
        for border in style.element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)
    normal = doc.styles["Normal"].paragraph_format
    normal.space_after = Pt(6)
    normal.line_spacing = 1.07
    for name in ("Heading 1", "Heading 2"):
        style = doc.styles[name]
        style.font.bold = True
        style.paragraph_format.space_before = Pt(9)
        style.paragraph_format.space_after = Pt(4)
    doc.styles["Title"].paragraph_format.space_after = Pt(7)
    doc.styles["Caption"].paragraph_format.space_before = Pt(3)
    doc.styles["Caption"].paragraph_format.space_after = Pt(5)
    doc.styles["Caption"].font.bold = False
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    number = OxmlElement("w:fldSimple")
    number.set(qn("w:instr"), "PAGE")
    footer._p.append(number)


def caption(doc, label, value, above=False):
    paragraph = doc.add_paragraph(style="Caption")
    paragraph.add_run(label + "  ").bold = True
    paragraph.add_run(value)
    paragraph.paragraph_format.keep_with_next = above


def table(doc, rows, widths):
    assert len(rows[0]) == len(widths) and abs(sum(widths) - 6.7) < 0.001
    result = doc.add_table(rows=0, cols=len(widths))
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    result.autofit = False
    for column, width in zip(result.columns, widths):
        column.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement("w:" + side)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:color"), "D9D9D9")
        edge.set(qn("w:sz"), "4")
        borders.append(edge)
    result._tbl.tblPr.append(borders)
    for idx, values in enumerate(rows):
        row = result.add_row()
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if idx == 0:
            row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for col, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT if col == 0 else WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_before = para.paragraph_format.space_after = Pt(3)
            para.paragraph_format.line_spacing = 1.0
            para.paragraph_format.keep_with_next = idx < len(rows) - 1
            run = para.add_run(value)
            run.font.size = Pt(9.5)
            run.bold = idx == 0
            if idx == 0:
                fill = OxmlElement("w:shd")
                fill.set(qn("w:fill"), "E9EEF3")
                cell._tc.get_or_add_tcPr().append(fill)


def diagram():
    canvas = Image.new("RGB", (1620, 430), "white")
    draw = ImageDraw.Draw(canvas)
    fontdir = Path("C:/Windows/Fonts")
    title = ImageFont.truetype(str(fontdir / "arialbd.ttf"), 33)
    body = ImageFont.truetype(str(fontdir / "arial.ttf"), 28)
    navy, teal, gray = "#204669", "#1B615E", "#52606D"

    def box(x, y, w, h, label, description, color, fill):
        draw.rounded_rectangle((x, y, x+w, y+h), radius=13, fill=fill, outline=color, width=3)
        draw.text((x+19, y+11), label, font=title, fill=color)
        draw.multiline_text((x+19, y+53), description, font=body, fill="#20252B", spacing=4)

    def arrow(x1, y, x2, color=gray):
        draw.line((x1, y, x2, y), fill=color, width=4)
        draw.polygon(((x2, y), (x2-13, y-8), (x2-13, y+8)), fill=color)

    box(7, 14, 337, 104, "Sequence", "Frozen ESM-2 residues", navy, "#EAF0F7")
    box(7, 136, 337, 132, "Structure", "AFDB coordinates\nFrozen GearNet features", navy, "#EAF0F7")
    box(7, 290, 337, 132, "Confidence", "Aligned, off or shuffled\nresidue pLDDT", teal, "#E9F4F1")
    box(409, 132, 343, 133, "Graph attention", "Edge geometry and\nconfidence prior", navy, "#EAF0F7")
    box(818, 132, 348, 133, "Residue fusion", "Sequence plus gated\nstructure update", navy, "#EAF0F7")
    box(1232, 132, 382, 133, "GO prediction", "Eight pooled summaries\nplus sequence head", navy, "#EAF0F7")
    arrow(344, 198, 407)
    arrow(752, 198, 816)
    arrow(1166, 198, 1230)
    draw.line((344, 66, 1418, 66, 1418, 129), fill=gray, width=4)
    draw.line((991, 66, 991, 129), fill=gray, width=4)
    draw.polygon(((991, 130), (983, 117), (999, 117)), fill=gray)
    draw.polygon(((1418, 130), (1410, 117), (1426, 117)), fill=gray)
    draw.line((380, 66, 380, 165), fill=gray, width=4)
    arrow(380, 165, 407)
    draw.line((344, 338, 991, 338, 991, 268), fill=teal, width=4)
    draw.line((577, 338, 577, 268), fill=teal, width=4)
    for x in (577, 991):
        draw.polygon(((x, 267), (x-8, 281), (x+8, 281)), fill=teal)
    stream = BytesIO()
    canvas.save(stream, format="PNG")
    stream.seek(0)
    return stream


def add_text(doc, text):
    after_table = doc._body._body[-2].tag == qn("w:tbl")
    for idx, paragraph in enumerate(text.strip().split("\n\n")):
        new = doc.add_paragraph(paragraph.strip())
        if idx == 0 and after_table:
            new.paragraph_format.space_before = Pt(6)


def picture(doc, path, width):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path) if isinstance(path, Path) else path, width=Inches(width))


def build(output):
    archive = {b: record(name) for b, name in ARCHIVED_FILES.items()}
    temporal = {b: record(f"{b}_uax_residue_temporal.json") for b in BRANCHES}
    second = {b: record(f"{b}_uax_benefit_experts.json") for b in BRANCHES}
    routing = {b: record(f"{b}_uax_benefit_prediction_extended.json") for b in BRANCHES}
    doc = Document()
    format_doc(doc)
    title = "Testing AlphaFold Confidence in Sequence and Structure Fusion for Protein Function Prediction"
    doc.core_properties.title = title
    doc.core_properties.author = "Ayaan Jain; Chua Chin Heng Matthew; Khanh Lee"
    title_para = doc.add_paragraph(title, "Title")
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author = doc.add_paragraph("Ayaan Jain, Chua Chin Heng Matthew, and Khanh Lee")
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.runs[0].bold = True
    author.paragraph_format.space_after = Pt(1)
    affiliation = doc.add_paragraph("Yong Loo Lin School of Medicine, National University of Singapore")
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Abstract", 1)
    add_text(doc, """
AlphaFold supplies a per-residue estimate of local structural accuracy. We tested whether this signal improves Gene Ontology (GO) prediction when protein sequence and predicted structure are fused. Our model combines frozen ESM-2 and GearNet residue representations and uses predicted local distance difference test (pLDDT) scores in spatial attention and a residue fusion gate. We compared sequence alone with matched fusion models using aligned, disabled, or shuffled confidence. Across two seeds on archived splits, aligned pLDDT changed maximum F-score relative to disabled confidence by −0.11 and −0.30 percentage points for biological process, +0.14 and −0.25 for cellular component, and +0.17 and +0.06 for molecular function. In a temporal human holdout, all paired fixed-threshold F1 intervals for this comparison included zero. Models fitted on validation proteins to predict which proteins would benefit from pLDDT achieved held-out AUROC of 0.501–0.522 using confidence summaries and length. No tested routing rule exceeded a fixed blend at its validation-selected operating point. These experiments show that local structural confidence has limited value as a general-purpose fusion signal in the tested architectures and cohorts.
""")
    keywords = doc.add_paragraph()
    keywords.add_run("Index terms  ").bold = True
    keywords.add_run("Protein function prediction, Gene Ontology, AlphaFold, structural confidence, multimodal fusion.")

    doc.add_heading("I. Introduction", 1)
    add_text(doc, """
The Gene Ontology organizes protein annotations into biological process (BPO), cellular component (CCO), and molecular function (MFO) [1]. Sequence predictors such as DeepGOPlus combine learned sequence features with homology transfer [2]. DeepFRI uses residue contact graphs and sequence-derived features [3], while Struct2GO and GAT-GO use predicted structures in graph models [4, 5]. Recent approaches include TAWFN score fusion, MAEF-GO multistage feature fusion, and MEGA-GO multiscale graph processing [6–8]. ProtNote and FunBind extend functional annotation through language or multimodal representations [9, 10].

AlphaFold makes predicted structures broadly available and assigns each residue a pLDDT score on a 0–100 scale [11, 12]. We use this estimate of local coordinate reliability as an operational proxy for structural epistemic confidence. It is distinct from uncertainty in GO labels, confidence in relative domain placement, and the incremental predictive value of structure. A confident fold may encode information already available to a sequence model. Conversely, a segment with low structural confidence can still carry functional sequence information.

We test whether explicit pLDDT adds value when sequence and structure are already available, whether its residue-level assignment matters, and whether we can predict the proteins for which it helps. We use paired controls, a date-based holdout, and an analysis of prediction changes when confidence is perturbed with weights held fixed. The contribution is an empirical evaluation of confidence as a fusion input.
""")

    doc.add_heading("II. Methods", 1)
    doc.add_heading("A. Cohorts and structure provenance", 2)
    add_text(doc, """
The archived cohort starts from the Struct2GO-derived training, validation, and test partitions [4]. We remove exact amino-acid sequence matches across partitions, prioritizing training and then validation. We retain proteins with exact agreement between benchmark sequences, reviewed human UniProt sequences, and AlphaFold Database v6 coordinates. After filtering, the BPO, CCO, and MFO training sets contain 7,221, 7,679, and 6,516 proteins, the validation sets contain 2,015, 2,194, and 1,860, and the test sets contain 1,044, 1,098, and 937.

The temporal cohort uses human experimental GO annotations from the January 17, 2024 and July 26, 2026 releases [1]. We use older records for training and validation and evaluate proteins whose eligible annotations for the branch are dated after the cutoff and whose sequences are absent from the older eligible branch records and all original benchmark partitions. We close reference labels over is_a and part_of relations using the older ontology and retain branch terms with at least 25 training proteins. Verified structural coverage leaves 68 BPO, 141 CCO, and 75 MFO test proteins. Some proteins occur in multiple branches. The two cohorts use different label sets. This is an annotation-time holdout using retrospective sequence and structure features.

We use the AlphaFold DB F1 model_v6 PDB record per matched accession and extract residue pLDDT from its Cα B-factor field. Each contact graph and confidence vector come from the same record. This protocol uses one released prediction per protein. Alternate predictions can change both contacts and pLDDT, and between-model variance is outside the scope of these data.
""")
    doc.add_heading("B. Sequence and structure fusion", 2)
    add_text(doc, """
We freeze the ESM-2 150M residue encoder and the released multiview-contrast GearNet-Edge encoder [13, 14]. We subtract the local-only GearNet representation from the full-graph representation at each residue and project the difference into the sequence feature space. The fusion graph connects residues up to two sequence positions apart, including self loops, and adds up to 16 neighbors within 10 Å at sequence separation greater than two. Spatial descriptors include distance, direction in a residue-local frame, orientation, and sequence separation.

Two four-head graph attention layers update the structural stream. For a spatial edge i → j, let pᵢ and pⱼ denote residue pLDDT divided by 100. We set rᵢⱼ = exp{−λ max[0, (0.9 − min(pᵢ, pⱼ))/0.2]}. Local edges have rᵢⱼ = 1. Each head adds log rᵢⱼ to its geometry-adjusted logit before softmax and multiplies the normalized message by rᵢⱼ. The second factor allows low reliability to reduce absolute message strength. These operations run in the fusion head, after GearNet encoding.

Let sᵢ be the projected ESM-2 state and gᵢ the resulting graph state. We form zᵢ = sᵢ + aᵢ ⊙ (gᵢ − sᵢ), where the sigmoid gate aᵢ uses sᵢ, gᵢ, and qᵢ = λ(pᵢ − 0.7), with an additive 2qᵢ in the gate logit. The head pools zᵢ into eight learned summaries and reads them out with GO term-specific weights. A separate classifier on mean-pooled ESM-2 contributes base logits to which we add 0.25 times the pooled head logits. Figure 1 shows where confidence enters. Encoder weights and internal message passing stay fixed.
""")
    picture(doc, diagram(), 6.66)
    caption(doc, "Fig. 1.", "Confidence enters spatial attention and the residue fusion gate. The disabled-confidence arm sets edge reliability to one and the gate’s confidence input to zero. The shuffled arm permutes pLDDT within each protein before both operations.")

    controls_heading = doc.add_heading("C. Controls and evaluation", 2)
    controls_heading.paragraph_format.page_break_before = True
    add_text(doc, """
The sequence-only arm bypasses structural fusion. The disabled, aligned, and shuffled arms share the architecture and structural inputs. Shuffling preserves each protein’s pLDDT distribution while breaking residue alignment. A validation-only geometry control permutes spatial endpoints and associated features, including the GearNet representations, within directed degree classes while retaining sequence-local edges. It tests the value of correctly placed spatial information within the candidate screen.

We used MFO validation to select a candidate and confidence strength by the aligned model’s Fmax margin over the stronger of the disabled and shuffled controls. We applied the selected configuration (λ = 0.20) across branches and cohorts. We trained the prediction head with binary cross-entropy, AdamW at learning rate 0.001 and weight decay 0.0001, width 96, and a 16-epoch cap. We selected checkpoints by Fmax with patience five and calibrated thresholds on a separate validation subset. A BPO audit found one exact repeat crossing these subsets. We grouped repeated sequences and reran BPO seed 1.

We report protein-centric Fmax from a threshold sweep of 0.01 to 1.00 and fixed F1 at each model’s validation-selected threshold. Precision is averaged over proteins with predicted labels and recall over proteins with reference labels. Paired 95% protein bootstrap intervals for F1 differences condition on fitted models and thresholds. In the perturbation analysis we change the confidence input at inference while retaining the aligned model’s weights and threshold. This tests dependence on the input under an intervention, without assigning functional importance to attention weights [15].
""")
    doc.add_heading("D. Predicting benefit before seeing labels", 2)
    add_text(doc, """
Using a second matched archived seed, we fit a ridge regression model (penalty 10) to predict the aligned model’s individual-protein F1 minus that of the disabled model. We split the calibration proteins into a fitting subset and an operating-point subset, grouping repeated sequences. We froze the expert thresholds defining the target on the fitting subset. Predictors used log length, mean and standard deviation of pLDDT, fractions below 50 and 70 and at least 90, and mean minimum endpoint confidence for spatial contacts. We compared length alone, prediction entropy and disagreement, combined inputs, shuffled confidence metadata, and exploratory shallow boosted trees.

On the operating-point subset we selected a cutoff for switching between the two experts and a final GO decision threshold. The comparator is a constant probability blend of the same two experts, tuned on the same subset. We measured test AUROC among proteins with positive or negative F1 change and evaluated the policy on all test proteins. This analysis is exploratory because it uses one expert seed and previously examined archived test cohorts.
""")

    doc.add_heading("III. Results", 1)
    doc.add_heading("A. Structural features and aligned confidence", 2)
    add_text(doc, """
In the MFO candidate screen, the selected GearNet model reached Fmax 0.7577 with aligned confidence, 0.7563 with confidence disabled, and 0.7556 with shuffled confidence. Sequence alone reached 0.7414 and randomized geometry reached 0.7583. The latter result limits attribution of the validation gain to spatial organization. The estimated confidence effect must therefore be read against both the disabled and shuffled controls.
""")
    caption(doc, "Table I.", "Seed 1 test performance. Model columns report Fmax. The right column reports aligned minus disabled fixed F1 in percentage points, with paired 95% protein bootstrap intervals.", above=True)
    rows = [["Test cohort", "n", "Sequence", "Disabled", "Aligned", "Shuffled", "Δ fixed F1 [95% CI]"]]
    rows += [metric_row(f"Archived {name}", archive[b]) for b, name in BRANCHES.items()]
    rows += [metric_row(f"Temporal {name}", temporal[b]) for b, name in BRANCHES.items()]
    table(doc, rows, [1.17, 0.45, 0.73, 0.72, 0.69, 0.70, 2.24])
    add_text(doc, """
On the archived MFO test, fusion with disabled confidence exceeded sequence alone in Fmax by 0.94 points. Enabling pLDDT added 0.17 points to Fmax, while the paired fixed F1 estimate was −0.15 points. For BPO, disabled fusion also exceeded the sequence baseline in Fmax and aligned confidence slightly reduced that score. For CCO, the three real-structure fusion variants and sequence alone were within 0.22 Fmax points. At the validation-selected threshold, aligned versus shuffled confidence improved archived CCO F1 by 0.39 points [0.01, 0.78] in this seed. This isolated estimate comes from several exploratory contrasts.

In the temporal MFO cohort, aligned fusion recovered 1.58 Fmax points over disabled fusion and reached 0.8160, the same rounded score as sequence alone. The fixed F1 difference relative to disabled fusion was +0.76 points [−1.85, +3.21], and relative to sequence alone it was +1.06 [−1.88, +3.82]. Temporal BPO favored sequence alone in Fmax. Temporal CCO favored disabled fusion in Fmax while its fixed F1 contrast between aligned and disabled fusion had a positive point estimate. All temporal aligned-versus-disabled intervals included zero.
""")
    doc.add_heading("B. Sensitivity across seeds and fusion designs", 2)
    rows = [["Branch", "Seed 1 ΔFmax", "Seed 2 ΔFmax", "Ridge AUROC [95% CI]", "Tree AUROC"]]
    for b, label in BRANCHES.items():
        deltas = []
        for run in (archive[b], second[b]):
            arms = by_arm(run)
            deltas.append(100 * (arms["aligned"]["test_fmax"] - arms["neutral"]["test_fmax"]))
        preds = routing[b]["benefit_predictors"]
        linear = preds["confidence_and_length"]["test_discrimination"]
        nonlinear = preds["confidence_tree"]["test_discrimination"]
        ci = linear["auroc_protein_bootstrap_95"]
        rows.append([label, *(f"{d:+.2f}" for d in deltas),
                     f"{linear['auroc_help_vs_harm']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                     f"{nonlinear['auroc_help_vs_harm']:.3f}"])
    caption(doc, "Table II.", "Archived change in Fmax from enabling pLDDT, in percentage points. AUROC is for predicting individual benefit from confidence summaries and length using seed 2.", above=True)
    table(doc, rows, [0.79, 1.13, 1.13, 2.26, 1.39])
    add_text(doc, """
Only MFO had a positive Fmax difference in both archived seeds, with gains below 0.2 points. In the fitted seed 1 model, replacing aligned pLDDT with shuffled pLDDT at inference changed fixed F1 by at most 0.15 points in either cohort. All six paired intervals for this perturbation included zero. Thus the fixed model’s predictions showed little sensitivity to residue alignment under this intervention.

On the larger archived splits used for complementary experiments, a compact graph encoder crossed an additive pLDDT attention prior with either static fusion or a mean-pLDDT gate. Across three seeds per branch, the prior changed mean Fmax by at most 0.00008 in absolute value, while the gate on ordinary attention added 0.0015, 0.0012, and 0.0059 for BPO, CCO, and MFO. A post-encoder confidence blend of adapted TAWFN blocks on MFO changed Fmax from 0.6720 to 0.6721 and fixed F1 from 0.6711 to 0.6709. The adaptation used narrower blocks and projected pooled SeqVec in place of native residue embeddings. On these larger splits, retrained full-width DeepGOPlus reached Fmax 0.5988, 0.7354, and 0.7807, and the retrained DeepFRI molecular-function configuration reached 0.6369. These reference values apply to their own input coverage and are not pooled with Table I.
""")

    routing_heading = doc.add_heading("C. Predicting which proteins benefit", 2)
    routing_heading.paragraph_format.page_break_before = True
    add_text(doc, """
The primary benefit predictors were near chance in each branch (Table II). Adding prediction entropy and disagreement gave ridge AUROC 0.524 for BPO, 0.488 for CCO, and 0.494 for MFO. The exploratory boosted tree reached 0.548 for BPO with all features and 0.542 for MFO with confidence and length. These small ranking differences did not translate into an advantage for selecting an expert.

The static blends achieved test fixed F1 of 0.5673, 0.7420, and 0.7675 for BPO, CCO, and MFO. The strongest fitted routing rules using unshuffled inputs reached 0.5580, 0.7394, and 0.7586, respectively. For MFO the confidence-and-length rules chose the aligned expert for every test protein. Mean pLDDT strata also showed no monotonic pattern. In MFO the aligned-minus-disabled fixed F1 estimate was −1.10 points below 70, +0.61 from 70 to below 90, and −0.25 at 90 or above. All three subgroup intervals included zero.
""")
    picture(doc, RESULTS / "uax_benefit_prediction_roc.png", 6.66)
    caption(doc, "Fig. 2.", "Test ROC curves for distinguishing positive from negative individual-protein F1 changes in the archived seed 2 models. Ties are excluded. The dashed line indicates chance discrimination. BP, CC, and MF refer to BPO, CCO, and MFO.")

    doc.add_heading("IV. Discussion", 1)
    add_text(doc, """
The experiments separate the contribution of a structural stream from the contribution of pLDDT inside that stream. In archived MFO the disabled model improved Fmax over the sequence model, with little additional change after enabling confidence. In temporal MFO the ordering of the disabled and sequence models reversed, and aligned confidence restored Fmax to the sequence model’s level. The CCO and BPO contrasts further show that the direction of the effect depends on the branch and evaluation threshold.

One possible explanation is that pLDDT scores the local geometry available to the model, while a GO label depends on whether the geometry at those residues adds discriminative information beyond sequence. A confidence prior also downweights groups of contacts according to coordinate reliability irrespective of their relationship to the label. The weak response to shuffling in the fitted model constrains how much residue assignment influenced these predictions. The experiments do not identify functional residues or resolve the biological mechanism behind branch differences. The post-encoder design isolates where the intervention takes place and can consume another encoder’s aligned residue vectors without modifying its layers. The experiments establish performance only for the tested implementations.

Exact sequence filtering leaves possible homology and pretraining overlap. The temporal cohorts are small, restricted to reviewed human proteins with verified predicted structures, and differ from the archived cohorts in annotation and label coverage. Fmax and fixed-threshold F1 answer different questions, and the protein bootstrap reflects sampling variation conditional on the fitted models. The benefit predictors used a single archived seed and require external confirmation. The fixed blend combines separately trained models, so its advantage can include ensembling effects and cannot be attributed specifically to confidence. We used one predicted structure per protein and did not measure uncertainty across alternative conformations or model runs.
""")
    conclusion_heading = doc.add_heading("V. Conclusion", 1)
    conclusion_heading.paragraph_format.page_break_before = True
    add_text(doc, """
Across two archived seeds and a temporal annotation holdout, pLDDT produced small or cohort-dependent changes in matched sequence–structure fusion. A slight archived MFO Fmax gain repeated across seeds, while the temporal fixed-threshold intervals remained compatible with benefit or harm. Confidence summaries and model outputs did not yield a routing rule that improved on a fixed blend. These results support evaluating confidence-conditioned models against matched disabled and shuffled controls before interpreting confidence as a guide to function prediction.
""")

    doc.add_heading("Data and code", 1)
    add_text(doc, """
The study uses Struct2GO [4], AlphaFold DB v6 [12], and human GO releases dated January 17, 2024 and July 26, 2026 (release.geneontology.org). REPRODUCE.md lists the scripts, run commands, and checkpoint information. The JSON experiment records contain the settings and paired results.
""")
    doc.add_heading("Acknowledgments", 1)
    add_text(doc, "The authors used OpenAI Codex and ChatGPT for assistance with software development, analysis, and manuscript preparation.")
    doc.add_heading("References", 1)
    refs = [
        "Gene Ontology Consortium, “The Gene Ontology resource: enriching a GOld mine,” Nucleic Acids Res., vol. 49, pp. D325–D334, 2021. doi: 10.1093/nar/gkaa1113.",
        "M. Kulmanov and R. Hoehndorf, “DeepGOPlus: improved protein function prediction from sequence,” Bioinformatics, vol. 36, pp. 422–429, 2020. doi: 10.1093/bioinformatics/btz595.",
        "V. Gligorijević et al., “Structure-based protein function prediction using graph convolutional networks,” Nat. Commun., vol. 12, art. 3168, 2021. doi: 10.1038/s41467-021-23303-9.",
        "P. Jiao et al., “Struct2GO: protein function prediction based on graph pooling algorithm and AlphaFold2 structure information,” Bioinformatics, vol. 39, btad637, 2023. doi: 10.1093/bioinformatics/btad637.",
        "B. Lai and J. Xu, “Accurate protein function prediction via graph attention networks with predicted structure information,” Brief. Bioinform., vol. 23, bbab502, 2022. doi: 10.1093/bib/bbab502.",
        "L. Meng and X. Wang, “TAWFN: a deep learning framework for protein function prediction,” Bioinformatics, vol. 40, btae571, 2024. doi: 10.1093/bioinformatics/btae571.",
        "M. Liu et al., “Multistage attention-based extraction and fusion of protein sequence and structural features for protein function prediction,” Bioinformatics, vol. 41, btaf374, 2025. doi: 10.1093/bioinformatics/btaf374.",
        "Y. Lee et al., “MEGA-GO: functions prediction of diverse protein sequence length using Multi-scalE Graph Adaptive neural network,” Bioinformatics, vol. 41, btaf032, 2025. doi: 10.1093/bioinformatics/btaf032.",
        "S. Char et al., “ProtNote: a multimodal method for protein–function annotation,” Bioinformatics, vol. 41, btaf170, 2025. doi: 10.1093/bioinformatics/btaf170.",
        "F. Boadu, Y. Wang, and J. Cheng, “A unified multimodal model for generalizable zero-shot and supervised protein function prediction,” Bioinformatics, vol. 42, btag356, 2026. doi: 10.1093/bioinformatics/btag356.",
        "J. Jumper et al., “Highly accurate protein structure prediction with AlphaFold,” Nature, vol. 596, pp. 583–589, 2021. doi: 10.1038/s41586-021-03819-2.",
        "M. Varadi et al., “AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences,” Nucleic Acids Res., vol. 52, pp. D368–D375, 2024. doi: 10.1093/nar/gkad1011.",
        "Z. Lin et al., “Evolutionary-scale prediction of atomic-level protein structure with a language model,” Science, vol. 379, pp. 1123–1130, 2023. doi: 10.1126/science.ade2574.",
        "Z. Zhang et al., “Protein representation learning by geometric structure pretraining,” in Proc. ICLR, 2023. arXiv:2203.06125.",
        "S. Jain and B. C. Wallace, “Attention is not explanation,” in Proc. NAACL-HLT, pp. 3543–3556, 2019. doi: 10.18653/v1/N19-1357.",
    ]
    for idx, ref in enumerate(refs, 1):
        para = doc.add_paragraph(f"[{idx}] {ref}")
        para.paragraph_format.keep_together = True
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.left_indent = Inches(0.24)
        para.paragraph_format.first_line_indent = Inches(-0.24)
        para.runs[0].font.size = Pt(9.5)
    doc.save(output)


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve())
