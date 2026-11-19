"""Build the binding-site manuscript from the frozen two-seed result."""

from io import BytesIO
import json
from pathlib import Path
from statistics import mean

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image, ImageDraw, ImageFont

from build_confidence_paper import add_text, caption, format_doc, picture, table


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "binding_confident_sites_complete.json"
OUTPUT = ROOT / "submission" / "Confidence_Guided_Binding_Site_Prediction.docx"
TITLE = "Sequence and Structure Fusion for Ligand Binding Site Prediction with AlphaFold Confidence"


def diagram():
    image = Image.new("RGB", (1560, 362), "white")
    draw = ImageDraw.Draw(image)
    fonts = Path("C:/Windows/Fonts")
    heading = ImageFont.truetype(str(fonts / "arialbd.ttf"), 31)
    body = ImageFont.truetype(str(fonts / "arial.ttf"), 27)
    blue, teal, gray = "#264D70", "#225F59", "#495B6D"

    def box(x, y, w, h, title, description, color=blue, fill="#ECF2F8"):
        draw.rounded_rectangle((x, y, x+w, y+h), 13, fill=fill, outline=color, width=3)
        draw.text((x+18, y+10), title, font=heading, fill=color)
        draw.multiline_text((x+18, y+49), description, font=body, fill="#21282E", spacing=4)

    def arrow(x1, y, x2, color=gray):
        draw.line((x1, y, x2, y), fill=color, width=4)
        draw.polygon(((x2, y), (x2-13, y-8), (x2-13, y+8)), fill=color)

    box(8, 8, 430, 125, "Residue representations", "ESM-2 sequence features\nGearNet and SaProt geometry")
    box(8, 152, 430, 125, "Template alignments", "DIAMOND and 3Di searches\nTraining labels only")
    box(490, 8, 490, 125, "Learned residue scores", "Spatial attention and gated\nstructural correction")
    box(490, 152, 490, 125, "Transferred evidence", "Votes from chemically compatible\naligned residues")
    box(1080, 78, 467, 130, "Ranked binding residues", "Add sequence votes, then\nstructural votes")
    box(8, 296, 430, 58, "AlphaFold pLDDT", "", teal, "#EAF5F1")
    arrow(438, 72, 489)
    arrow(438, 214, 489)
    draw.line((980, 72, 1027, 72, 1027, 143, 1080, 143), width=4, fill=gray)
    draw.line((980, 214, 1027, 214, 1027, 143), width=4, fill=gray)
    draw.polygon(((1080, 143), (1067, 135), (1067, 151)), fill=gray)
    draw.line((438, 325, 466, 325, 466, 106, 488, 106), fill=teal, width=4)
    arrow(466, 247, 488, teal)
    arrow(466, 106, 488, teal)
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def new_page(doc, title):
    paragraph = doc.add_heading(title, 1)
    paragraph.paragraph_format.page_break_before = True


def aggregate(result, variant, transfer=None):
    return [result["metrics"][f"{variant}_{seed}"] if transfer is None
            else result["residue_transfer"][f"{variant}_{seed}"][transfer] for seed in (0, 1)]


def main():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    doc = Document()
    format_doc(doc)
    doc.core_properties.title = TITLE
    doc.core_properties.author = "Ayaan Jain; Chua Chin Heng Matthew; Khanh Lee"
    title = doc.add_paragraph(TITLE, "Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author = doc.add_paragraph("Ayaan Jain, Chua Chin Heng Matthew, and Khanh Lee")
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.runs[0].bold = True
    author.paragraph_format.space_after = Pt(1)
    affiliation = doc.add_paragraph("Yong Loo Lin School of Medicine, National University of Singapore")
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Abstract", 1)
    add_text(doc, """
We combined protein sequence embeddings, AlphaFold geometry, and structural template transfer to predict ligand-binding residues. We used predicted local distance difference test scores (pLDDT) to weight spatial messages and template votes. On 262 human proteins held out by sequence similarity group, the full model increased macro average precision from 0.5316 for a sequence classifier with sequence template transfer to 0.5729. The gain across two seeds was 4.13 percentage points (paired 95% group bootstrap interval +1.18 to +6.18). The full model differed from its confidence-disabled control by −0.40 points (−1.49 to +0.69) and from a within-protein confidence shuffle by +1.34 points (+0.04 to +2.40). The largest gain came from combining learned geometry and structurally aligned templates.
""")
    keywords = doc.add_paragraph()
    keywords.add_run("Index terms  ").bold = True
    keywords.add_run("Ligand binding, protein language models, AlphaFold, structural confidence, template transfer.")

    doc.add_heading("I. Introduction", 1)
    add_text(doc, """
A protein sequence can contain conserved coordinating motifs, and predicted structures can place distant sequence positions together in a binding pocket. GraphBind and GPSite use learned geometric features to identify binding residues [1, 2]. Template transfer supplies a complementary form of evidence by mapping annotations from corresponding positions in known proteins.

Predicted coordinates vary in reliability across residues. AlphaFold supplies pLDDT as a per-residue estimate of local structural accuracy [3, 4]. We use pLDDT as a proxy for the reliability of a predicted structural input. A high score alone gives no evidence that a site binds a ligand. We test whether learned geometric evidence and template evidence improve a sequence reference, and whether correctly assigning pLDDT changes prediction quality when the structural inputs are held fixed.
""")

    doc.add_heading("II. Dataset and evaluation split", 1)
    add_text(doc, """
We retrieved reviewed human UniProtKB entries with positional binding or active-site annotations from release 2026_04 [5]. The binding task uses individual residues annotated with the Binding site feature and pools ligand types. We matched canonical sequences exactly to the corresponding AlphaFold DB v6 model and excluded duplicate sequences. Training used all exact-position binding annotations. Validation and test used experimentally supported annotations (ECO:0000269) as positives. Other known annotated positions and ambiguous ranges were masked in evaluation. Remaining residues served as operational negatives. They can include binding residues without annotations.

We used DIAMOND to group similar proteins [6]. An edge joined sequences if an alignment at E ≤ 0.001 had at least 30% identity and both aligned spans were at least the smaller of 100 residues or 70% of the shorter protein’s length. We assigned entire connected components to train, validation, and test. The binding task retained 2,574 training proteins, 147 validation proteins, and 262 test proteins in 130 groups. The test labels include 2,070 experimentally supported binding residues.
""")

    new_page(doc, "III. Methods")
    doc.add_heading("A. Learned sequence and geometry", 2)
    add_text(doc, """
We froze ESM-2 150M, GearNet-Edge, and SaProt 35M [7–9]. The sequence arm projects ESM-2 residue vectors into 96 channels and produces one logit per residue. For the structural arm, we subtract the local-only GearNet representation from its full-graph representation and concatenate it with SaProt embeddings generated from 3Di states with amino acid identity masked. We project the structural representation to the same width. The graph connects each residue to up to 16 Cα neighbors within 10 Å separated by more than two sequence positions. Distance, direction in a local frame, relative orientation, and sequence separation provide edge features.

The sequence vector queries projected structural neighbor vectors. In the aligned-confidence variant, pLDDT divided by 100 gives qᵢ. For edge i to j we use reliability rᵢⱼ = exp{−0.7 max[0, (0.9 − min(qᵢ, qⱼ))/0.2]}. We add log rᵢⱼ to the edge logit before softmax and multiply each resulting attention weight by rᵢⱼ. A sigmoid gate takes the sequence vector, structural update, and centered confidence cᵢ = qᵢ − 0.7 as inputs and adds 2cᵢ to the learned gate logit. The gate scales a structural correction to the sequence logit. The disabled control uses rᵢⱼ = 1 and cᵢ = 0. The shuffled control permutes pLDDT within each protein.
""")
    picture(doc, diagram(), 6.65)
    caption(doc, "Fig. 1.", "The learned predictor and template transfers supply residue-level evidence. Confidence weights spatial messages and the reliability of aligned template residues.")

    doc.add_heading("B. Residue template transfer", 2)
    add_text(doc, """
We aligned validation and test queries only against training proteins. We used DIAMOND 2.2.8 for amino acid alignments and MMseqs2 release 18 with the Foldseek 3Di substitution matrix for structural-alphabet alignments [6, 10, 11]. The 3Di search used sensitivity 7.5, 120 prefilter candidates, gap costs 10 and 1, and a minimum alignment length of 15. We retained structural hits at E ≤ 0.01. We mapped gapped alignments back to residue indices in the matched canonical sequences.

For a hit with E-value E and bit score B, let u = clip(−log₁₀ E / 30, 0.01, 1) and w = u min(B/200, 1)². Each aligned template residue contributes to the denominator of a weighted vote. It contributes a positive vote when annotated and chemically compatible with the query residue. Compatibility means identity or membership in the same predefined amino acid group (DE, KR, ST, FWY, NQ, or ILVM). In the confidence-conditioned structural arm, both the vote weight and the support u are multiplied by clip[(min(qᵢ, qⱼ) − 0.50)/0.40, 0.05, 1]. The transfer score t is the positive vote fraction multiplied by the strongest local support. It is zero for residues without aligned support.

For predicted residue score p, the update is p + α(1 − p)t. We apply sequence template votes first and structural template votes second. We choose each α from 0, 0.25, 0.5, and 1 by validation average precision. The sequence reference uses the same sequence classifier and sequence transfer step.
""")

    new_page(doc, "IV. Results")
    add_text(doc, """
We trained the neural heads with AdamW (learning rate 0.001, weight decay 0.0001) for up to 14 epochs, using binary cross-entropy and positive class weight equal to the square root of the training negative-to-positive ratio. Checkpoints maximize validation macro average precision (AP) with patience five. We report macro AP averaged over proteins and then over two seeds. Micro AP pools residues within each seed. Top-three recall averages the fraction of each protein’s annotated residues recovered in its three highest-scoring predictions. Paired intervals use 2,000 bootstrap resamples of test sequence groups and condition on the two fitted seeds.
""")
    labels = [
        ("Sequence classifier", "sequence", None),
        ("Sequence classifier and sequence templates", "sequence", "sequence_transfer"),
        ("Sequence model and all templates with confidence", "sequence", "real"),
        ("Learned sequence and structure with confidence", "aligned", None),
        ("Full sequence and structure with confidence", "aligned", "real"),
        ("Full sequence and structure with confidence disabled", "neutral", "off"),
        ("Full sequence and structure with shuffled confidence", "shuffled", "shuffled"),
    ]
    rows = [["Model", "Macro AP\nseed 0", "Macro AP\nseed 1", "Mean\nmacro AP", "Mean\nmicro AP", "Mean top-3\nrecall"]]
    for label, variant, transfer in labels:
        data = aggregate(result, variant, transfer)
        scores = [row["macro_ap"] for row in data]
        rows.append([label, *(f"{v:.4f}" for v in (*scores, mean(scores),
                    mean(row["micro_ap"] for row in data), mean(row["recall_top3"] for row in data)))])
    caption(doc, "Table I.", "Ligand-binding residue prediction on the same 262 proteins for every model.", above=True)
    table(doc, rows, [2.65, 0.75, 0.75, 0.78, 0.78, 0.99])
    add_text(doc, """
Adding confidence-weighted structural templates to the sequence classifier with sequence templates improved macro AP by 2.22 points (95% interval +0.65 to +3.68). The full confidence-conditioned system added learned spatial features as well and improved macro AP by 4.13 points over the sequence reference. The seed-level improvements were 4.59 and 3.67 points. Its mean micro AP increased from 0.4907 to 0.5524, and its top-three recall increased from 0.2997 to 0.3165.
""")
    rows = [["Paired comparison", "Macro AP change [95% interval]"]]
    for label, key in [
        ("Full aligned vs sequence classifier with sequence templates", "aligned_real_minus_sequence_sequence_transfer"),
        ("Full aligned vs full confidence disabled", "aligned_real_minus_neutral_off"),
        ("Full aligned vs full confidence shuffled", "aligned_real_minus_shuffled_shuffled"),
        ("Structural template confidence vs disabled with same sequence head", "sequence_real_minus_sequence_off"),
    ]:
        contrast = result["paired_transfer_macro_ap"][key]
        lo, hi = contrast["bootstrap_95"]
        rows.append([label, f"{100 * contrast['macro_ap_delta']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"])
    caption(doc, "Table II.", "Paired changes in macro AP in percentage points.", above=True)
    table(doc, rows, [4.20, 2.50])
    add_text(doc, """
Aligned confidence scored 1.34 points above shuffled confidence in the full system, with an interval just above zero. Disabling confidence produced the highest mean macro AP among the full systems, 0.5770. Shuffling tests whether residue assignment matters when confidence weighting is present. Disabling confidence tests the value of introducing the weighting.
""")

    new_page(doc, "V. Discussion")
    add_text(doc, """
The template experiment isolates structural correspondences while holding the sequence classifier fixed. The full model adds a learned spatial correction and achieved the largest gain over sequence-based evidence. pLDDT can dampen unreliable geometry, but binding depends on ligand chemistry, conformational state, and evolutionary conservation. The two controls place the clearest gain on the geometric inputs themselves. The shuffled contrast is exploratory and its interval is unadjusted for multiple comparisons.

This evaluation uses one human cohort, one predicted structure per accession, and two training seeds. Similarity grouping can leave remote homologs or overlap with pretraining data. The task recovers known binding annotations, which incompletely describe the residues that bind each ligand. We evaluated neural-only pilots before adding template transfer and selected its weights on validation. An independent, ligand-specific test would establish how well the gain transfers.
""")
    doc.add_heading("Data and code", 1)
    add_text(doc, "REPRODUCE.md records the UniProtKB query, AlphaFold DB inputs, and run commands. The two-seed record is binding_confident_sites_complete.json.")
    doc.add_heading("Acknowledgments", 1)
    add_text(doc, "The authors used OpenAI Codex and ChatGPT for assistance with software development, analysis, and manuscript preparation.")
    doc.add_heading("References", 1)
    references = [
        "Y. Xia et al., GraphBind. Nucleic Acids Res. 49, e51 (2021). doi: 10.1093/nar/gkab044.",
        "Q. Yuan et al., Genome-scale annotation of protein binding sites via language model and geometric deep learning. eLife 13, RP93695 (2024). doi: 10.7554/eLife.93695.",
        "J. Jumper et al., Highly accurate protein structure prediction with AlphaFold. Nature 596, 583–589 (2021). doi: 10.1038/s41586-021-03819-2.",
        "M. Varadi et al., AlphaFold Protein Structure Database in 2024. Nucleic Acids Res. 52, D368–D375 (2024). doi: 10.1093/nar/gkad1011.",
        "S. Ahmad et al., The UniProt website API. Nucleic Acids Res. 53, W547–W553 (2025). doi: 10.1093/nar/gkaf394.",
        "B. Buchfink, K. Reuter, and H.-G. Drost, Sensitive protein alignments at tree-of-life scale using DIAMOND. Nat. Methods 18, 366–368 (2021). doi: 10.1038/s41592-021-01101-x.",
        "Z. Lin et al., Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123–1130 (2023). doi: 10.1126/science.ade2574.",
        "Z. Zhang et al., Protein representation learning by geometric structure pretraining. ICLR (2023). arXiv:2203.06125.",
        "J. Su et al., SaProt: protein language modeling with structure-aware vocabulary. ICLR (2024). doi: 10.1101/2023.10.01.560349.",
        "M. Steinegger and J. Söding, MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat. Biotechnol. 35, 1026–1028 (2017). doi: 10.1038/nbt.3988.",
        "M. van Kempen et al., Fast and accurate protein structure search with Foldseek. Nat. Biotechnol. 42, 243–246 (2024). doi: 10.1038/s41587-023-01773-0.",
    ]
    for i, reference in enumerate(references, 1):
        para = doc.add_paragraph(f"[{i}] {reference}")
        para.paragraph_format.keep_together = True
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.left_indent = Inches(0.24)
        para.paragraph_format.first_line_indent = Inches(-0.24)
        para.runs[0].font.size = Pt(9.5)
    doc.save(OUTPUT)


if __name__ == "__main__":
    main()
