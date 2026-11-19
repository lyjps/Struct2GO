# Exact reference changes

The comparison below was extracted directly from the supplied and corrected Word files.
Only citation markers and bibliography entries differ in document text.
Before and after blocks reproduce the wording, punctuation, and numbers in those files.

## Citation changes in manuscript order

### A. The Critical Role of Protein Function Prediction in Modern Biology

**Before**

```text
Proteins are among the most complex structures in biological systems. [1] They are large strings of amino acids that form a three-dimensional structure and carry out a wide array of critical biological functions and are essential components of nearly every organic process, controlling a variety of functions. [2] Therefore, protein function prediction (PFP) remains a fundamental, persistent challenge in contemporary computational biology, serving as a critical upstream task for understanding disease mechanisms, identifying drug targets, and unraveling complex cellular pathways. [3] This is furthered by its use for predicting which individual residues bind ligands. [A]                                                                                     
```

**After**

```text
Proteins are among the most complex structures in biological systems. [1] They are large strings of amino acids that form a three-dimensional structure and carry out a wide array of critical biological functions and are essential components of nearly every organic process, controlling a variety of functions. [1] Therefore, protein function prediction (PFP) remains a fundamental, persistent challenge in contemporary computational biology, serving as a critical upstream task for understanding disease mechanisms, identifying drug targets, and unraveling complex cellular pathways. [2] This is furthered by its use for predicting which individual residues bind ligands. [3]                                                                                     
```

### B. The introduction of protein structure dependant models

**Before**

```text
Although various technologies have been created to experimentally solve protein structures, such as X-ray Crystallography, Fluorescence Spectroscopy, and Circular Dichroism,[7] these methods are often time consuming and expensive.[8] Currently, the more than 100 000 proteins that have been discovered in the Universal Protein (UniProt) database with standard functional annotations, account for only 0.1% of the proteins in the whole database. [9] Therefore, newer mathematical and computational techniques have been developed for protein structure prediction. Of these techniques, machine learning based modelling using deep learning has been especially successful. The model Alphafold2, developed by Google Deepmind has been accurate to a large degree, and can correctly predict structures down to the atomic-level. [10] It uses deep learning techniques, namely transformer inspired architecture, and the biological basis of evolution in proteins to predict new structures. [8]
```

**After**

```text
Although various technologies have been created to experimentally solve protein structures, such as X-ray Crystallography, Fluorescence Spectroscopy, and Circular Dichroism,[7] these methods are often time consuming and expensive.[8] Currently, the more than 100 000 proteins that have been discovered in the Universal Protein (UniProt) database with standard functional annotations, account for only 0.1% of the proteins in the whole database. [9] Therefore, newer mathematical and computational techniques have been developed for protein structure prediction. Of these techniques, machine learning based modelling using deep learning has been especially successful. The model Alphafold2, developed by Google Deepmind has been accurate to a large degree, and can correctly predict structures down to the atomic-level. [10] It uses deep learning techniques, namely transformer inspired architecture, and the biological basis of evolution in proteins to predict new structures. [10]
```

### B. The introduction of protein structure dependant models

**Before**

```text
This breakthrough has facilitated the development of structure-based PFP models that utilize Graph Neural Networks (GNNs) to model the spatial relationships between amino acid residues. [11] These GNN models were then further improved upon with the introduction of Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs). [6] As well as TAWFN which combined predictions from a sequence CNN and a graph network. [A] GCNs and GATs have thus become the standard for processing this 3D data by representing residues as nodes and their spatial proximity as edges. [9] For example, the HEAL model utilizes hierarchical graph transformers to capture both local residue environments and global protein topology, while Struct2GO employs self-attention graph pooling to aggregate features for whole-protein classification. [13][14] Similarly, PFP models such as GAT-GO which utilises Graph Attention Networks (GAT), are built upon attention frameworks that enable them to learn relationships between a protein’s sequence and structure and its function, improving prediction accuracy. [15] 
```

**After**

```text
This breakthrough has facilitated the development of structure-based PFP models that utilize Graph Neural Networks (GNNs) to model the spatial relationships between amino acid residues. [6] These GNN models were then further improved upon with the introduction of Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs). [6] As well as TAWFN which combined predictions from a sequence CNN and a graph network. [11] GCNs and GATs have thus become the standard for processing this 3D data by representing residues as nodes and their spatial proximity as edges. [12] For example, the HEAL model utilizes hierarchical graph transformers to capture both local residue environments and global protein topology, while Struct2GO employs self-attention graph pooling to aggregate features for whole-protein classification. [12, 13] Similarly, PFP models such as GAT-GO which utilises Graph Attention Networks (GAT), are built upon attention frameworks that enable them to learn relationships between a protein’s sequence and structure and its function, improving prediction accuracy. [14] 
```

### C. Limitations of Current Solutions

**Before**

```text
Current state-of-the-art multimodal models, exemplified by approaches like Struct2GO and ESM-GearNet, have established the improvement achieved by fusing contextual sequence embeddings, often derived from large Protein Language Models (PLMs), with geometric features encoded by Graph Neural Networks (GNNs). [9][16] This fusion paradigm leverages diverse information streams to address the complexity in classifying proteins across the hierarchical framework of the Gene Ontology (GO). [17] For instance, Struct2GO extracts structural features from AlphaFold2-derived contact graphs using GNNs and hierarchical self-attention pooling, subsequently integrating them with sequence features, thus improving the accuracy of predictions. [18]
```

**After**

```text
Current state-of-the-art multimodal models, exemplified by approaches like Struct2GO and ESM-GearNet, have established the improvement achieved by fusing contextual sequence embeddings, often derived from large Protein Language Models (PLMs), with geometric features encoded by Graph Neural Networks (GNNs). [12, 15] This fusion paradigm leverages diverse information streams to address the complexity in classifying proteins across the hierarchical framework of the Gene Ontology (GO). [16] For instance, Struct2GO extracts structural features from AlphaFold2-derived contact graphs using GNNs and hierarchical self-attention pooling, subsequently integrating them with sequence features, thus improving the accuracy of predictions. [12]
```

### C. Limitations of Current Solutions

**Before**

```text
A fundamental limitation however, persists within these multimodal approaches that utilise sequence and structure representations. Most operate under the assumption of uniform reliability across the predicted structural input. [19] While AlphaFold2 structures are impressive, they are heterogeneous in reliability because different protein fold predictions, or structure predictions, differ in the confidence of their prediction. [20] However, most multi-modal approaches to PFP treat protein structures as being homogenous in reliability. [20] Furthermore, many methods treat protein structures as being equal in confidence to protein sequence data. [21]
```

**After**

```text
A fundamental limitation however, persists within these multimodal approaches that utilise sequence and structure representations. Most operate under the assumption of uniform reliability across the predicted structural input. [17] While AlphaFold2 structures are impressive, they are heterogeneous in reliability because different protein fold predictions, or structure predictions, differ in the confidence of their prediction. [18] However, most multi-modal approaches to PFP treat protein structures as being homogenous in reliability. [18] Furthermore, many methods treat protein structures as being equal in confidence to protein sequence data. 
```

### C. Limitations of Current Solutions

**Before**

```text
This can be detrimental as features derived from unreliable low-confidence structural regions introduce systematic noise into the learned representations, with there being very few useful structural regions that have low reliability scores [22][23], therefore compromising the model’s overall predictive accuracy and generalization capabilities. This issue is particularly pronounced when dealing with proteins containing intrinsically disordered regions. [24] To investigate the challenge of unreliable structural input, the Uncertainty-Aware Cross Attention Fusion (UAX) framework is proposed. [25]
```

**After**

```text
This can be detrimental as features derived from unreliable low-confidence structural regions introduce systematic noise into the learned representations, with there being very few useful structural regions that have low reliability scores [19, 20], therefore compromising the model’s overall predictive accuracy and generalization capabilities. This issue is particularly pronounced when dealing with proteins containing intrinsically disordered regions. [21] To investigate the challenge of unreliable structural input, the Uncertainty-Aware Cross Attention Fusion (UAX) framework is proposed. 
```

### D. Explorations using epistemic confidence

**Before**

```text
This article proposes the novel usage of the UAX architecture, processing multi-modal protein data with a confidence guided cross-attention fusion mechanism for feature fusion. By implementing these measures, we hope to investigate the accuracy of protein function predictions through the introduction of epistemic confidence and enhanced fusion and function interaction through attention, and introduce improvements for GO branches by including this. [26]
```

**After**

```text
This article proposes the novel usage of the UAX architecture, processing multi-modal protein data with a confidence guided cross-attention fusion mechanism for feature fusion. By implementing these measures, we hope to investigate the accuracy of protein function predictions through the introduction of epistemic confidence and enhanced fusion and function interaction through attention, and introduce improvements for GO branches by including this. 
```

### D. Explorations using epistemic confidence

**Before**

```text
UAX is built upon the principle that the model must “know what it does not know”. The original model, AlphaFold2, usefully provides an ideal metric for this purpose, a per-residue confidence metric known as pLDDT (predicted Local Distance Difference Test) the epistemic confidence score. [8] This score, ranging from 0 to 100, directly correlates with the local accuracy of the predicted structure.
```

**After**

```text
UAX is built upon the principle that the model must “know what it does not know”. The original model, AlphaFold2, usefully provides an ideal metric for this purpose, a per-residue confidence metric known as pLDDT (predicted Local Distance Difference Test) the epistemic confidence score. [10] This score, ranging from 0 to 100, directly correlates with the local accuracy of the predicted structure.
```

### D. Explorations using epistemic confidence

**Before**

```text
To control the influence of unreliable structural inputs, we incorporate pLDDT by first normalizing it to the range 0 to 1, by dividing it by 100, and it is then used as a continuous confidence signal. Instead of using pLDDT as a binary mask, our architecture treats it as a soft reliability gate, enabling smooth attenuation of uncertain structural features. [27] This allows predicted structures which have lower pLDDT, to contribute proportionally less to the final fused representation while preserving high-confidence structural cues. Then we hold structural features and template alignments available while comparing real, disabled, and shuffled confidence.
```

**After**

```text
To control the influence of unreliable structural inputs, we incorporate pLDDT by first normalizing it to the range 0 to 1, by dividing it by 100, and it is then used as a continuous confidence signal. Instead of using pLDDT as a binary mask, our architecture treats it as a soft reliability gate, enabling smooth attenuation of uncertain structural features. [22] This allows predicted structures which have lower pLDDT, to contribute proportionally less to the final fused representation while preserving high-confidence structural cues. Then we hold structural features and template alignments available while comparing real, disabled, and shuffled confidence.
```

### B. Sequence Embedding Branch

**Before**

```text
A large jump in the Natural Language Processing field has been beneficial for obtaining valuable data from protein sequences and therefore the dataset assembled in Struct2GO utilised the SeqVec model for extracting semantic information related to protein function from sequential information. [28] These embeddings provide a fixed-length numerical representation of the entire protein sequence, capturing deep evolutionary context. We project the sequence features of a protein from a 1×1024 matrix to a 1×48 matrix.
```

**After**

```text
A large jump in the Natural Language Processing field has been beneficial for obtaining valuable data from protein sequences and therefore the dataset assembled in Struct2GO utilised the SeqVec model for extracting semantic information related to protein function from sequential information. [23] These embeddings provide a fixed-length numerical representation of the entire protein sequence, capturing deep evolutionary context. We project the sequence features of a protein from a 1×1024 matrix to a 1×48 matrix.
```

### C. Independent Confidence Pathways for GO

**Before**

```text
This branch utilizes a graph G = (V, E) where nodes vi represent residues and edges eij represent spatial proximity within a 10Å distance threshold Using archived Struct2GO contact graphs. [10] 
```

**After**

```text
This branch utilizes a graph G = (V, E) where nodes vi represent residues and edges eij represent spatial proximity within a 10Å distance threshold Using archived Struct2GO contact graphs. [12] 
```

### C. Independent Confidence Pathways for GO

**Before**

```text
Even still without explicit uncertainty modeling, standard attention mechanisms assign comparable importance to interactions regardless of reliability, resulting in inaccurate predictions driven by noisy structure. [29]
```

**After**

```text
Even still without explicit uncertainty modeling, standard attention mechanisms assign comparable importance to interactions regardless of reliability, resulting in inaccurate predictions driven by noisy structure. [24]
```

### C. Independent Confidence Pathways for GO

**Before**

```text
To address this issue, we propose confidence-weighted graph attention, which incorporates pLDDT into the computation by utilizing Graph Attention Networks, allowing the model to modulate information flow based on structural reliability. [30]
```

**After**

```text
To address this issue, we propose confidence-weighted graph attention, which incorporates pLDDT into the computation by utilizing Graph Attention Networks, allowing the model to modulate information flow based on structural reliability. [25]
```

### D. GO Baselines and Backbone Tests

**Before**

```text
We train the compact graph models with width 48 for up to eight epochs using AdamW. On the same filtered splits we retrain full-width DeepGOPlus with training-only DIAMOND transfer and select its blend on validation. [A, A]
```

**After**

```text
We train the compact graph models with width 48 for up to eight epochs using AdamW. On the same filtered splits we retrain full-width DeepGOPlus with training-only DIAMOND transfer and select its blend on validation. [5, 26]
```

### D. GO Baselines and Backbone Tests

**Before**

```text
For the backbone test, we adapt TAWFN CNN and adaptive GCN blocks to the available amino acid identities, projected pooled SeqVec features and stored contacts, with width 32. [A] We train the two predictors on MFO, freeze them and alternate between the static or pLDDT score gate on validation predictions. 
```

**After**

```text
For the backbone test, we adapt TAWFN CNN and adaptive GCN blocks to the available amino acid identities, projected pooled SeqVec features and stored contacts, with width 32. [11] We train the two predictors on MFO, freeze them and alternate between the static or pLDDT score gate on validation predictions. 
```

### E. Multi-Stage Convolution and Pooling

**Before**

```text
The architecture then employs a series of Convolution Pools to capture both adaptive and aggregated neighborhood information. Each block follows a three-stage GCN refinement pipeline to capture multi-scale neighbourhood information. The refined features are then passed through a Self-Attention Graph Pooling layer. [31] This layer computes node scores using two independent graph convolution layers, retaining a subset of nodes based on a predefined pooling ratio. We then combine scores using an arithmetic mean to ensure numerical stability and to avoid over-supression of nodes:
```

**After**

```text
The architecture then employs a series of Convolution Pools to capture both adaptive and aggregated neighborhood information. Each block follows a three-stage GCN refinement pipeline to capture multi-scale neighbourhood information. The refined features are then passed through a Self-Attention Graph Pooling layer. [27] This layer computes node scores using two independent graph convolution layers, retaining a subset of nodes based on a predefined pooling ratio. We then combine scores using an arithmetic mean to ensure numerical stability and to avoid over-supression of nodes:
```

### 1) Confidence-Gated Structure-Sequence Fusion: 

**Before**

```text
Due to the differing reliabilities of protein structure predictions, we argue that structure should be trusted proportionally to its predicted reliability, and that sequence information should compensate when structure is uncertain. This motivates a fusion mechanism that adaptively balances structure and sequence based on global structural confidence, thus being able to dynamically adjust the degree to which it utilises structure and sequence representations. [32][33]
```

**After**

```text
Due to the differing reliabilities of protein structure predictions, we argue that structure should be trusted proportionally to its predicted reliability, and that sequence information should compensate when structure is uncertain. This motivates a fusion mechanism that adaptively balances structure and sequence based on global structural confidence, thus being able to dynamically adjust the degree to which it utilises structure and sequence representations. [22, 28]
```

### 1) Confidence-Gated Structure-Sequence Fusion: 

**Before**

```text
This scalar confidence summary is passed through a learnable linear transformation followed by a sigmoid activation. [34]
```

**After**

```text
This scalar confidence summary is passed through a learnable linear transformation followed by a sigmoid activation. [29]
```

### 1) Confidence-Gated Structure-Sequence Fusion: 

**Before**

```text
This formulation enables the model to dynamically tune the extent to which structure representations and sequence representations are included in the final fused representation, dependent on the reliability of the predicted protein structure. [32] Furthermore, although pLDDT is used as an input signal, the mapping from confidence to modality weighting is learned end-to-end via Wgate. This crucially allows the model to discover non-linear relationships between structural confidence and functional relevance, rather than enforcing a fixed or manually tuned weighting scheme.
```

**After**

```text
This formulation enables the model to dynamically tune the extent to which structure representations and sequence representations are included in the final fused representation, dependent on the reliability of the predicted protein structure. [22] Furthermore, although pLDDT is used as an input signal, the mapping from confidence to modality weighting is learned end-to-end via Wgate. This crucially allows the model to discover non-linear relationships between structural confidence and functional relevance, rather than enforcing a fixed or manually tuned weighting scheme.
```

### 1) Confidence-Gated Structure-Sequence Fusion: 

**Before**

```text
We deliberately use the global mean pLDDT as a coarse but stable summary of structural reliability to avoid sensitivity to local noise and to introduce minimal inductive bias, while still capturing the overall trustworthiness of the predicted structure. This provides an improvement to previous models that equally weighted structural and sequential representations as will be shown in the results section. [35][36]
```

**After**

```text
We deliberately use the global mean pLDDT as a coarse but stable summary of structural reliability to avoid sensitivity to local noise and to introduce minimal inductive bias, while still capturing the overall trustworthiness of the predicted structure. This provides an improvement to previous models that equally weighted structural and sequential representations as will be shown in the results section. [30]
```

### 2) Cross-Attention Interactive Fusion: 

**Before**

```text
While global confidence gating provides a useful, coarse, mechanism for adjusting the overall influence of predicted structure, it does not thoroughly capture which structural features are relevant to which specific sequence contexts. [37] Since the relevance of structural information for proteins varies across residues, sequence-structure fusion in proteins is inherently context-dependent. [38] Fixed fusion strategies are rarely able to model such cross-model dependencies, forcing downstream layers to disentangle interactions implicitly. [39]
```

**After**

```text
While global confidence gating provides a useful, coarse, mechanism for adjusting the overall influence of predicted structure, it does not thoroughly capture which structural features are relevant to which specific sequence contexts. [31] Since the relevance of structural information for proteins varies across residues, sequence-structure fusion in proteins is inherently context-dependent. [15] Fixed fusion strategies are rarely able to model such cross-model dependencies, forcing downstream layers to disentangle interactions implicitly. [32]
```

### 2) Cross-Attention Interactive Fusion: 

**Before**

```text
We therefore propose a cross-attention framework where one modality may attend to the other, such as structure representations attending to sequence-derived embeddings. [40] This design allows the querying , such as sequence, to determine which features of the other modality, such as structure, should be incorporated. Cross-attention therefore enables explicit, content-driven interaction between modalities, rather than relying on implicit mixing through concatena    tion or global weighting. It acts as a fine-grained tool for interactive fusion.
```

**After**

```text
We therefore propose a cross-attention framework where one modality may attend to the other, such as structure representations attending to sequence-derived embeddings. [33] This design allows the querying , such as sequence, to determine which features of the other modality, such as structure, should be incorporated. Cross-attention therefore enables explicit, content-driven interaction between modalities, rather than relying on implicit mixing through concatena    tion or global weighting. It acts as a fine-grained tool for interactive fusion.
```

### III. RESULTS AND DISCUSSION

**Before**

```text
To validate the improvements of the UAX framework we conducted quantitative analysis with ablation studies on two PFP backbones, Struct2GO and GAT-GO papers. Specifically, we compare baseline Struct2GO, a convolution-based model, with Struct2GO that uses UAX and baseline GAT-GO. We then repeat this for DeepGOPlus [A], DeepFRI [A], and TAWFN [A]. Therefore we are able to isolate and compare the improvements of our architecture across these differing backbones.
```

**After**

```text
To validate the improvements of the UAX framework we conducted quantitative analysis with ablation studies on two PFP backbones, Struct2GO and GAT-GO papers. Specifically, we compare baseline Struct2GO, a convolution-based model, with Struct2GO that uses UAX and baseline GAT-GO. We then repeat this for DeepGOPlus [5], DeepFRI [6], and TAWFN [11]. Therefore we are able to isolate and compare the improvements of our architecture across these differing backbones.
```

### A. Conclusion

**Before**

```text
Before concluding, it is important to note that localized geometric precision does not directly equate to functional utility. This is because the GO edge prior changes relative attention coefficients, and homogenous reliability metrics across spatial neighborhoods yield virtually unchanged normalized attention distributions. Furthermore, attenuating low-confidence structural segments poses the risk of discarding critical functional signals, which presents a possible explanation for the modest and variable empirical shifts observed. Our controlled ablations use the uncertainty aware architecture to re-evaluate the performance drops in BPO and CCO previously noted in epistemic confidence architectures [A] by incorporating a unified encoding architecture and evaluation framework. Under these conditions, the performance improvement in MFO stems from the late confidence gating mechanism.
```

**After**

```text
Before concluding, it is important to note that localized geometric precision does not directly equate to functional utility. This is because the GO edge prior changes relative attention coefficients, and homogenous reliability metrics across spatial neighborhoods yield virtually unchanged normalized attention distributions. Furthermore, attenuating low-confidence structural segments poses the risk of discarding critical functional signals, which presents a possible explanation for the modest and variable empirical shifts observed. Our controlled ablations use the uncertainty aware architecture to re-evaluate the performance drops in BPO and CCO previously noted in epistemic confidence architectures  by incorporating a unified encoding architecture and evaluation framework. Under these conditions, the performance improvement in MFO stems from the late confidence gating mechanism.
```

### B. Future Work

**Before**

```text
Individually ranked AlphaFold2 monomer predictions typically generate multiple ranked conformed candidates that exhibit distinct local confidence profiles and contact topographies. [26] In this study, we selected a single public structure per target, before the measurement of inter-model variance. Future investigations that alternate structural conformations should establish deterministic filtering criteria prior to testing, such as anchoring baseline comparisons to the top-ranked prediction. For ensemble architectures, each structural candidate should be mapped to the canonical sequence, constructing distinct contact graphs from respective spatial coordinates while maintaining paired per-residue pLDDT profiles to assess prediction variance across models. Averaged ensemble outputs can subsequently be benchmarked against single-structure baselines.
```

**After**

```text
Individually ranked AlphaFold2 monomer predictions typically generate multiple ranked conformed candidates that exhibit distinct local confidence profiles and contact topographies. [34] In this study, we selected a single public structure per target, before the measurement of inter-model variance. Future investigations that alternate structural conformations should establish deterministic filtering criteria prior to testing, such as anchoring baseline comparisons to the top-ranked prediction. For ensemble architectures, each structural candidate should be mapped to the canonical sequence, constructing distinct contact graphs from respective spatial coordinates while maintaining paired per-residue pLDDT profiles to assess prediction variance across models. Averaged ensemble outputs can subsequently be benchmarked against single-structure baselines.
```

## Bibliography entries in original order

### Original reference [1]

**Before**

```text
[1] Alberts et al., “The Shape and Structure of Proteins,” in Molecular Biology of the Cell, 4th ed., edited by B. Alberts et al. (Garland Science, New York, 2002), ch. 3. NCBI
```

**After**

```text
[1] B. Alberts et al., “The shape and structure of proteins,” in Molecular Biology of the Cell, 4th ed. New York, NY, USA: Garland Science, 2002, ch. 3. [Online]. Available: https://www.ncbi.nlm.nih.gov/books/NBK26830/
```

### Original reference [2]

**Before**

```text
[2] Zhu, Y., Chen, X., & Zhang, L. (2023). Cellular components: Proteins and their crucial role in cell function. Journal of Biological Research, 30(1), 12.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [3]

**Before**

```text
[3] Breda, Valadares, de Souza, and Garratt, “Protein Structure, Modelling and Applications,” in Bioinformatics in Tropical Disease Research: A Practical and Case-Study Approach, edited by A. Gruber et al. (National Center for Biotechnology Information, Bethesda, 2008), chap. A06.
```

**After**

```text
[2] A. Breda, N. F. Valadares, O. N. de Souza, and R. C. Garratt, “Protein structure, modelling and applications,” in Bioinformatics in Tropical Disease Research: A Practical and Case-Study Approach, A. Gruber et al., Eds. Bethesda, MD, USA: NCBI, 2008, ch. A06. [Online]. Available: https://www.ncbi.nlm.nih.gov/books/NBK6824/
```

### Original reference [4]

**Before**

```text
[4] Altschul, S. F., Gish, W., Miller, W., Myers, E. W., & Lipman, D. J. (1990). Basic local alignment search tool. Journal of molecular biology, 215(3), 403–410.
```

**After**

```text
[4] S. F. Altschul, W. Gish, W. Miller, E. W. Myers, and D. J. Lipman, “Basic local alignment search tool,” Journal of Molecular Biology, vol. 215, no. 3, pp. 403–410, 1990, doi: 10.1016/S0022-2836(05)80360-2.
```

### Original reference [5]

**Before**

```text
[5] Maxat Kulmanov, Robert Hoehndorf, (2019). DeepGOPlus: improved protein function prediction from sequence, Bioinformatics, Volume 36, Issue 2, January 2020, Pages 422–429.
```

**After**

```text
[5] M. Kulmanov and R. Hoehndorf, “DeepGOPlus: improved protein function prediction from sequence,” Bioinformatics, vol. 36, no. 2, pp. 422–429, 2020, doi: 10.1093/bioinformatics/btz595.
```

### Original reference [6]

**Before**

```text
[6] Gligorijević, V., Renfrew, P.D., Kosciolek, T. et al. Structure-based protein function prediction using graph convolutional networks. Nat Commun 12, 3168 (2021).
```

**After**

```text
[6] V. Gligorijević et al., “Structure-based protein function prediction using graph convolutional networks,” Nature Communications, vol. 12, art. 3168, 2021, doi: 10.1038/s41467-021-23303-9.
```

### Original reference [7]

**Before**

```text
[7] Berg, J. M., Tymoczko, J. L., & Stryer, L. (2015). Biochemistry (8th ed.). W.H. Freeman.
```

**After**

```text
[7] J. M. Berg, J. L. Tymoczko, G. J. Gatto, Jr., and L. Stryer, Biochemistry, 8th ed. New York, NY, USA: W. H. Freeman, 2015.
```

### Original reference [8]

**Before**

```text
[8] Schwede, T. (2013). Protein modeling: what’s next? Nature Methods, 10(1), 54–55.
```

**After**

```text
[8] T. Schwede, “Protein modeling: what happened to the ‘protein structure gap’?” Structure, vol. 21, no. 9, pp. 1531–1540, 2013, doi: 10.1016/j.str.2013.08.007.
```

### Original reference [9]

**Before**

```text
[9] The UniProt Consortium, “UniProt: the Universal Protein Knowledgebase in 2025,” Nucleic Acids Res. 53, D609–D617 (2025).
```

**After**

```text
[9] The UniProt Consortium, “UniProt: the Universal Protein Knowledgebase in 2025,” Nucleic Acids Research, vol. 53, no. D1, pp. D609–D617, 2025, doi: 10.1093/nar/gkae1010.
```

### Original reference [10]

**Before**

```text
[10] Z. Yang, X. Zeng, Y. Zhao, R. Chen,…, “AlphaFold2 and its applications in the fields of biology and medicine,” Signal Transduction and Targeted Therapy 8, 115 (2023).
```

**After**

```text
[10] J. Jumper et al., “Highly accurate protein structure prediction with AlphaFold,” Nature, vol. 596, pp. 583–589, 2021, doi: 10.1038/s41586-021-03819-2.
```

### Original reference [11]

**Before**

```text
[11] Ingraham, J., et al. (2019). Generative models for graph-based protein design. Advances in Neural Information Processing Systems (NeurIPS)
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [12]

**Before**

```text
[12] Illergård, K., Ardell, D. H., & Elofsson, A. (2009). Structure is three to ten times more conserved than sequence—A study of structural predictors in sequence alignments. Proteins: Structure, Function, and Bioinformatics
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [13]

**Before**

```text
[13] Zhu, Y., Chen, X., Zhang, L., & Liu, S. (2023). Hierarchical graph transformer with contrastive learning for protein function prediction. Bioinformatics, 39(7), btad396.
```

**After**

```text
[13] Z. Gu, X. Luo, J. Chen, M. Deng, and L. Lai, “Hierarchical graph transformer with contrastive learning for protein function prediction,” Bioinformatics, vol. 39, no. 7, art. btad410, 2023, doi: 10.1093/bioinformatics/btad410.
```

### Original reference [14]

**Before**

```text
[14] P. Jiao, B. Wang, X. Wang, B. Liu, Y. Wang, and J. Li, “Struct2GO: protein function prediction based on graph pooling algorithm and AlphaFold2 structure information,” Bioinformatics 39, btad637 (2023).
```

**After**

```text
[12] P. Jiao, B. Wang, X. Wang, B. Liu, Y. Wang, and J. Li, “Struct2GO: protein function prediction based on graph pooling algorithm and AlphaFold2 structure information,” Bioinformatics, vol. 39, no. 10, art. btad637, 2023, doi: 10.1093/bioinformatics/btad637.
```

### Original reference [15]

**Before**

```text
[15] Zhang, Y., Lin, J., Zhao, L., & Zheng, W. M. (2023). ProtFold: A dual-branch network for protein function prediction via sequence and structure. Bioinformatics, 39(Supplement_1), i397–i405.
```

**After**

```text
[14] B. Lai and J. Xu, “Accurate protein function prediction via graph attention networks with predicted structure information,” Briefings in Bioinformatics, vol. 23, no. 1, art. bbab502, 2022, doi: 10.1093/bib/bbab502.
```

### Original reference [16]

**Before**

```text
[16] Zhang, Z., Wang, C., Xu, M., Chenthamarakshan, V., Lozano, A.C., Das, P., & Tang, J. (2023). A Systematic Study of Joint Representation Learning on Protein Sequences and Structures.
```

**After**

```text
[15] Z. Zhang et al., “A systematic study of joint representation learning on protein sequences and structures,” arXiv:2303.06275, 2023. [Online]. Available: https://arxiv.org/abs/2303.06275
```

### Original reference [17]

**Before**

```text
[17] The Gene Ontology Consortium. (2021). The Gene Ontology resource: enriching a GOld mine. Nucleic Acids Research, 49(D1), D325–D334.
```

**After**

```text
[16] The Gene Ontology Consortium, “The Gene Ontology resource: enriching a GOld mine,” Nucleic Acids Research, vol. 49, no. D1, pp. D325–D334, 2021, doi: 10.1093/nar/gkaa1113.
```

### Original reference [18]

**Before**

```text
[18] Breda, Valadares, de Souza, and Garratt, “Protein Structure, Modelling and Applications,” in Bioinformatics in Tropical Disease Research: A Practical and Case-Study Approach.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [19]

**Before**

```text
[19] Akdel, M., Pires, D. E. V., Pardo, E. P., Jänes, J., Zalevsky, A. O. et al. (2022). A structural biology community assessment of AlphaFold2 applications. Nature Structural & Molecular Biology, 29(11), 1056–1067.
```

**After**

```text
[17] M. Akdel et al., “A structural biology community assessment of AlphaFold2 applications,” Nature Structural & Molecular Biology, vol. 29, pp. 1056–1067, 2022, doi: 10.1038/s41594-022-00849-w.
```

### Original reference [20]

**Before**

```text
[20] Williams, Christopher & Chen, Vincent & Richardson, David & Richardson, Jane. (2025). Categorizing prediction modes within low-pLDDT regions of AlphaFold 2 structures: near-predictive, pseudostructure and barbed wire. Acta Crystallographica Section D Structural Biology. 81.
```

**After**

```text
[18] C. J. Williams, V. B. Chen, D. C. Richardson, and J. S. Richardson, “Categorizing prediction modes within low-pLDDT regions of AlphaFold2 structures: near-predictive, pseudostructure and barbed wire,” Acta Crystallographica Section D: Structural Biology, vol. 81, no. 10, pp. 558–572, 2025, doi: 10.1107/S2059798325007843.
```

### Original reference [21]

**Before**

```text
[21] Ma, M., Liang, J., Lan, C., Yang, W., & Xu, Z. (2022). Multimodal learning with high-quality and noisy data: A survey. Information Fusion, 82, 1–16.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [22]

**Before**

```text
[22] Dunker, A. K., Silman, I., Uversky, V. N., Foster, D. M., & Sussman, J. L. (2008). The concept of an intrinsically disordered protein. FEBS Journal, 275(8), 1753–1764.
```

**After**

```text
[19] A. K. Dunker, I. Silman, V. N. Uversky, and J. L. Sussman, “Function and structure of inherently disordered proteins,” Current Opinion in Structural Biology, vol. 18, no. 6, pp. 756–764, 2008, doi: 10.1016/j.sbi.2008.10.002.
```

### Original reference [23]

**Before**

```text
[23] Ruff, K. M., & Pappu, R. V. (2021). AlphaFold and implications for intrinsically disordered proteins. Journal of Molecular Biology, 433(20), 167208.
```

**After**

```text
[20] K. M. Ruff and R. V. Pappu, “AlphaFold and implications for intrinsically disordered proteins,” Journal of Molecular Biology, vol. 433, no. 20, art. 167208, 2021, doi: 10.1016/j.jmb.2021.167208.
```

### Original reference [24]

**Before**

```text
[24] Tunyasuvunakool, K., Adler, J., Wu, Z. et al. Highly accurate protein structure prediction for the human proteome. Nature 596, 590–596 (2021).
```

**After**

```text
[21] K. Tunyasuvunakool et al., “Highly accurate protein structure prediction for the human proteome,” Nature, vol. 596, pp. 590–596, 2021, doi: 10.1038/s41586-021-03828-1.
```

### Original reference [25]

**Before**

```text
[25] Abdar, M., Pourpanah, F., Hussain, S., Rezazadegan, D., Liu, L., et al. (2021). A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion, 76, 243–297.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [26]

**Before**

```text
[26] Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? Advances in Neural Information Processing Systems, 30.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [27]

**Before**

```text
[27] Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language modeling with gated convolutional networks. International Conference on Machine Learning, 70, 933–941.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [28]

**Before**

```text
[28] Heinzinger, M., Elnaggar, A., Wang, Y., Dallago, C., Nechaev, D., Matthes, F., & Rost, B. (2019). Modeling aspects of the language of life through transfer-learning protein sequences. BMC bioinformatics, 20(1), 723.
```

**After**

```text
[23] M. Heinzinger et al., “Modeling aspects of the language of life through transfer-learning protein sequences,” BMC Bioinformatics, vol. 20, art. 723, 2019, doi: 10.1186/s12859-019-3220-8.
```

### Original reference [29]

**Before**

```text
[29] Brody, S., Alon, U., & Yahav, E. (2021). How Attentive are Graph Attention Networks? ArXiv, abs/2105.14491.
```

**After**

```text
[24] S. Brody, U. Alon, and E. Yahav, “How attentive are graph attention networks?” in Proc. International Conference on Learning Representations, 2022. [Online]. Available: https://arxiv.org/abs/2105.14491
```

### Original reference [30]

**Before**

```text
[30] Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2017). Graph Attention Networks. ArXiv, abs/1710.10903.
```

**After**

```text
[25] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph attention networks,” in Proc. International Conference on Learning Representations, 2018. [Online]. Available: https://arxiv.org/abs/1710.10903
```

### Original reference [31]

**Before**

```text
[31] Lee, J., Lee, I., & Kang, J. (2019). Self-attention graph pooling. International Conference on Machine Learning, 97, 3734–3743.
```

**After**

```text
[27] J. Lee, I. Lee, and J. Kang, “Self-attention graph pooling,” in Proc. 36th International Conference on Machine Learning, PMLR, vol. 97, pp. 3734–3743, 2019. [Online]. Available: https://proceedings.mlr.press/v97/lee19c.html
```

### Original reference [32]

**Before**

```text
[32] Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F.A. (2017). Gated Multimodal Units for Information Fusion. ArXiv, abs/1702.01992.
```

**After**

```text
[22] J. Arevalo, T. Solorio, M. Montes-y-Gómez, and F. A. González, “Gated multimodal units for information fusion,” arXiv:1702.01992, 2017. [Online]. Available: https://arxiv.org/abs/1702.01992
```

### Original reference [33]

**Before**

```text
[33] Wu, R., Wang, H., Chen, H., & Carneiro, G. (2024). Deep Multimodal Learning with Missing Modality: A Survey.
```

**After**

```text
[28] R. Wu, H. Wang, H.-T. Chen, and G. Carneiro, “Deep multimodal learning with missing modality: A survey,” arXiv:2409.07825, 2024. [Online]. Available: https://arxiv.org/abs/2409.07825
```

### Original reference [34]

**Before**

```text
[34] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
```

**After**

```text
[29] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical evaluation of gated recurrent neural networks on sequence modeling,” arXiv:1412.3555, 2014. [Online]. Available: https://arxiv.org/abs/1412.3555
```

### Original reference [35]

**Before**

```text
[35] Vielzeuf, V., Lechervy, A., Pateux, S., & Jurie, F. (2018). CentralNet: a multi-level approach for multimodal fusion. Proceedings of the European Conference on Computer Vision (ECCV) Workshops, 11129, 575–589.
```

**After**

```text
[30] V. Vielzeuf, A. Lechervy, S. Pateux, and F. Jurie, “CentralNet: a multilayer approach for multimodal fusion,” arXiv:1808.07275, 2018. [Online]. Available: https://arxiv.org/abs/1808.07275
```

### Original reference [36]

**Before**

```text
[36] Hu, G., Li, Q., Gao, J., & Wang, J. (2021). Deep learning for protein function prediction: methods, resources and challenges. Journal of Molecular Biology, 433(14), 167098.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [37]

**Before**

```text
[37] Chen, K., Tan, Z., & Liang, D. (2023). Multi-level fusion strategies for multimodal deep learning. Information Fusion, 93, 230–242.
```

**After**

```text
[31] F. Zhao, C. Zhang, and B. Geng, “Deep multimodal data fusion,” ACM Computing Surveys, vol. 56, no. 9, art. 216, pp. 1–36, 2024, doi: 10.1145/3649447.
```

### Original reference [38]

**Before**

```text
[38] Ingraham, J., Garg, V.K., Barzilay, R., & Jaakkola, T. (2019). Generative Models for Graph-Based Protein Design. DGS@ICLR.
```

**After**

Entry removed. Its in-text citation changes, where present, appear above.

### Original reference [39]

**Before**

```text
[39] Perez, E., Strub, F., Vries, H.D., Dumoulin, V., & Courville, A.C. (2017). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI Conference on Artificial Intelligence.
```

**After**

```text
[32] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conference on Artificial Intelligence, vol. 32, no. 1, pp. 3942–3951, 2018, doi: 10.1609/aaai.v32i1.11671.
```

### Original reference [40]

**Before**

```text
[40] Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., & Salakhutdinov, R. (2019). Multimodal transformer for unaligned multimodal language sequences. Association for Computational Linguistics.
```

**After**

```text
[33] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proc. 57th Annual Meeting of the Association for Computational Linguistics, pp. 6558–6569, 2019, doi: 10.18653/v1/P19-1656.
```

## Added bibliography entries

### New reference [3]

```text
[3] A. Gazizov, A. Lian, C. Goverde, J. Mou, S. Ovchinnikov, and N. F. Polizzi, “AF2BIND: predicting small-molecule binding sites using the pair representation of AlphaFold2,” Nature Methods, vol. 23, pp. 626–635, 2026, doi: 10.1038/s41592-026-03011-2.
```

### New reference [11]

```text
[11] L. Meng and X. Wang, “TAWFN: a deep learning framework for protein function prediction,” Bioinformatics, vol. 40, no. 10, art. btae571, 2024, doi: 10.1093/bioinformatics/btae571.
```

### New reference [26]

```text
[26] B. Buchfink, K. Reuter, and H.-G. Drost, “Sensitive protein alignments at tree-of-life scale using DIAMOND,” Nature Methods, vol. 18, pp. 366–368, 2021, doi: 10.1038/s41592-021-01101-x.
```

### New reference [34]

```text
[34] Google DeepMind, “AlphaFold: AlphaFold output,” GitHub repository documentation. Accessed: Nov. 15, 2026. [Online]. Available: https://github.com/google-deepmind/alphafold#alphafold-output
```

## Unresolved supporting evidence

The [A] after 'previously noted in epistemic confidence architectures' was removed. No verified external source was identified for the specific earlier BPO/CCO performance claim. The sentence itself was preserved.
Some remaining substantive claims require manuscript revision, including the current UniProt statistic, pLDDT as epistemic confidence, and the claimed lack of useful low-confidence regions.
