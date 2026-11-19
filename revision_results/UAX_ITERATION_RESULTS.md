# Confidence-guided contact attention and sequence–structure fusion

The experiments preserve two features of UAX. Residue pLDDT affects message passing on the contact graph, and a confidence-conditioned gate controls how structure contributes to the protein-level GO predictions. The molecular-function benchmark uses the archived graph and SeqVec inputs with exact sequence matches to earlier partitions excluded. The previous compact UAX reconstruction serves as the original-style comparison.

## Architecture changes

- Adding the logarithm of reliability lowers the attention logit of each uncertain contact. Reliability is a smooth function of the lower endpoint pLDDT. It also scales the message after softmax, so uniformly low reliability can reduce the amount of structural evidence passed. Self loops and edges up to two residues apart in sequence keep unit reliability.
- The sequence predictor produces base GO logits. The graph predicts a correction, and a separate gate for each GO term controls that correction using sequence and graph embeddings plus mean pLDDT, its standard deviation, and fractions below 70 and at least 90.
- The contextual variant adds 64 training-fitted principal components of frozen DeepFRI LSTM residue embeddings. The local-only control sees the same residue features and ignores edges spanning more than two sequence positions.
- The motif variant adds four sequence-conditioned graph pooling heads. The gate receives pLDDT averages under the same weights used to pool each of the four motif embeddings.

## Validation screen

All candidates used seed 0 and eight epochs at most. The first half of validation selected the epoch, and the second half produced the Fmax scores shown here. The neutral controls use the same architecture with confidence inputs disabled. The shuffled controls permute residues within each protein for contact confidence and reassign the global confidence profile across proteins.

| Configuration | Aligned pLDDT | Neutral | Shuffled | Aligned minus stronger control |
| --- | ---: | ---: | ---: | ---: |
| Residual gate with one-hot residues | 0.7063 | 0.7060 | 0.7073 | −0.0010 |
| Residual gate with contextual residues | 0.7098 | 0.7109 | 0.7097 | −0.0011 |
| Residual gate with contextual residues and motif confidence | 0.7083 | 0.7071 | 0.7088 | −0.0005 |

The motif configuration was selected by the largest margin over the stronger matched control. All three margins were negative. The original-style compact UAX and sequence-only controls reached 0.6802 and 0.7026, respectively, in the first screen. The contextual local-only and neutral full-graph controls reached 0.7109 and 0.7109 in the second screen.
