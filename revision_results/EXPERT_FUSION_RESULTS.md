# Homology aware expert fusion results

## test_f1_at_valid_threshold

| Expert mixture | BPO | CCO | MFO |
| --- | --- | --- | --- |
| DIAMOND transfer | 0.5521 | 0.6391 | 0.7404 |
| Compact CNN + DIAMOND | 0.5965 | 0.7205 | 0.7732 |
| Sequence experts, static | 0.5921 | 0.7331 | 0.7831 |
| Sequence experts, alignment gate | 0.5965 | 0.7405 | 0.7930 |
| Sequence and graph experts, static | 0.5921 | 0.7331 | 0.7830 |
| Sequence and graph experts, alignment gate | 0.5967 | 0.7404 | 0.7933 |
| Alignment + real pLDDT | 0.5958 | 0.7412 | 0.7935 |
| Alignment + shuffled pLDDT | 0.5962 | 0.7407 | 0.7931 |

## test_fmax

| Expert mixture | BPO | CCO | MFO |
| --- | --- | --- | --- |
| DIAMOND transfer | 0.5521 | 0.6396 | 0.7413 |
| Compact CNN + DIAMOND | 0.5965 | 0.7209 | 0.7732 |
| Sequence experts, static | 0.5927 | 0.7335 | 0.7838 |
| Sequence experts, alignment gate | 0.5976 | 0.7433 | 0.7938 |
| Sequence and graph experts, static | 0.5927 | 0.7335 | 0.7835 |
| Sequence and graph experts, alignment gate | 0.5976 | 0.7433 | 0.7938 |
| Alignment + real pLDDT | 0.5978 | 0.7441 | 0.7939 |
| Alignment + shuffled pLDDT | 0.5972 | 0.7434 | 0.7940 |

## test_micro_aupr

| Expert mixture | BPO | CCO | MFO |
| --- | --- | --- | --- |
| DIAMOND transfer | 0.4547 | 0.5552 | 0.7417 |
| Compact CNN + DIAMOND | 0.4941 | 0.6922 | 0.7864 |
| Sequence experts, static | 0.5443 | 0.7326 | 0.8220 |
| Sequence experts, alignment gate | 0.5606 | 0.7478 | 0.8381 |
| Sequence and graph experts, static | 0.5443 | 0.7326 | 0.8217 |
| Sequence and graph experts, alignment gate | 0.5607 | 0.7479 | 0.8382 |
| Alignment + real pLDDT | 0.5606 | 0.7485 | 0.8384 |
| Alignment + shuffled pLDDT | 0.5608 | 0.7482 | 0.8384 |

## Paired overall fixed-threshold F1 contrasts

| Comparison | BPO | CCO | MFO |
| --- | --- | --- | --- |
| Sequence gate vs. CNN + DIAMOND | +0.0000 [-0.0062, +0.0062] | +0.0200 [+0.0153, +0.0246] | +0.0199 [+0.0147, +0.0251] |
| Sequence gate vs. static mixture | +0.0045 [+0.0018, +0.0072] | +0.0074 [+0.0035, +0.0117] | +0.0100 [+0.0058, +0.0136] |
| Add graph expert | +0.0002 [-0.0003, +0.0006] | -0.0001 [-0.0005, +0.0003] | +0.0002 [-0.0001, +0.0006] |
| Add real pLDDT | -0.0009 [-0.0018, +0.0001] | +0.0008 [+0.0002, +0.0013] | +0.0002 [-0.0003, +0.0007] |
| Real vs. shuffled pLDDT | -0.0003 [-0.0010, +0.0003] | +0.0005 [+0.0000, +0.0009] | +0.0003 [+0.0000, +0.0007] |

Sequence routing versus the compact CNN and DIAMOND blend changed fixed-threshold F1 by BPO +0.0000 [-0.0062, +0.0062], CCO +0.0200 [+0.0153, +0.0246], MFO +0.0199 [+0.0147, +0.0251]. Adding the graph expert changed fixed-threshold F1 by BPO +0.0002 [-0.0003, +0.0006], CCO -0.0001 [-0.0005, +0.0003], MFO +0.0002 [-0.0001, +0.0006]. Using real instead of shuffled pLDDT changed fixed-threshold F1 by BPO -0.0003 [-0.0010, +0.0003], CCO +0.0005 [+0.0000, +0.0009], MFO +0.0003 [+0.0000, +0.0007]. The corresponding descriptive micro AUPR differences for sequence routing versus the compact blend were BPO +0.0664, CCO +0.0555, MFO +0.0517.

BPO (n = 834) had a graph expert effect of -0.0003 [-0.0008, +0.0003] and a real-versus-shuffled pLDDT effect of +0.0001 [-0.0007, +0.0009]. CCO (n = 887) had a graph expert effect of -0.0003 [-0.0008, +0.0002] and a real-versus-shuffled pLDDT effect of +0.0005 [-0.0001, +0.0010]. MFO (n = 715) had a graph expert effect of +0.0001 [-0.0003, +0.0006] and a real-versus-shuffled pLDDT effect of +0.0006 [+0.0001, +0.0012].

## Complete subgroup contrasts

| Branch | Contrast | Subset | n | Delta and 95% interval |
| --- | --- | --- | --- | --- |
| BPO | sequence_routed_minus_deepgoplus_control | all | 1198 | +0.0000 [-0.0062, +0.0062] |
| BPO | sequence_routed_minus_deepgoplus_control | no_close_hit | 834 | +0.0098 [+0.0001, +0.0201] |
| BPO | sequence_routed_minus_deepgoplus_control | close_hit | 364 | +0.0363 [+0.0258, +0.0469] |
| BPO | sequence_routed_minus_deepgoplus_control | no_detected_hit | 358 | +0.3004 [+0.2807, +0.3201] |
| BPO | sequence_routed_minus_sequence_static | all | 1198 | +0.0045 [+0.0018, +0.0072] |
| BPO | sequence_routed_minus_sequence_static | no_close_hit | 834 | +0.0012 [-0.0022, +0.0046] |
| BPO | sequence_routed_minus_sequence_static | close_hit | 364 | +0.0231 [+0.0174, +0.0288] |
| BPO | sequence_routed_minus_sequence_static | no_detected_hit | 358 | +0.0535 [+0.0440, +0.0636] |
| BPO | structure_routed_minus_sequence_routed | all | 1198 | +0.0002 [-0.0003, +0.0006] |
| BPO | structure_routed_minus_sequence_routed | no_close_hit | 834 | -0.0003 [-0.0008, +0.0003] |
| BPO | structure_routed_minus_sequence_routed | close_hit | 364 | +0.0009 [+0.0002, +0.0017] |
| BPO | structure_routed_minus_sequence_routed | no_detected_hit | 358 | -0.0013 [-0.0022, -0.0004] |
| BPO | confidence_routed_minus_structure_routed | all | 1198 | -0.0009 [-0.0018, +0.0001] |
| BPO | confidence_routed_minus_structure_routed | no_close_hit | 834 | +0.0000 [-0.0013, +0.0013] |
| BPO | confidence_routed_minus_structure_routed | close_hit | 364 | -0.0011 [-0.0026, +0.0004] |
| BPO | confidence_routed_minus_structure_routed | no_detected_hit | 358 | +0.0063 [+0.0039, +0.0086] |
| BPO | confidence_routed_minus_shuffled_routed | all | 1198 | -0.0003 [-0.0010, +0.0003] |
| BPO | confidence_routed_minus_shuffled_routed | no_close_hit | 834 | +0.0001 [-0.0007, +0.0009] |
| BPO | confidence_routed_minus_shuffled_routed | close_hit | 364 | -0.0005 [-0.0016, +0.0007] |
| BPO | confidence_routed_minus_shuffled_routed | no_detected_hit | 358 | +0.0023 [+0.0010, +0.0037] |
| BPO | structure_routed_minus_deepgoplus_control | all | 1198 | +0.0002 [-0.0060, +0.0063] |
| BPO | structure_routed_minus_deepgoplus_control | no_close_hit | 834 | +0.0095 [-0.0001, +0.0195] |
| BPO | structure_routed_minus_deepgoplus_control | close_hit | 364 | +0.0372 [+0.0264, +0.0482] |
| BPO | structure_routed_minus_deepgoplus_control | no_detected_hit | 358 | +0.2991 [+0.2795, +0.3188] |
| CCO | sequence_routed_minus_deepgoplus_control | all | 1278 | +0.0200 [+0.0153, +0.0246] |
| CCO | sequence_routed_minus_deepgoplus_control | no_close_hit | 887 | +0.0200 [+0.0135, +0.0269] |
| CCO | sequence_routed_minus_deepgoplus_control | close_hit | 391 | +0.0271 [+0.0198, +0.0342] |
| CCO | sequence_routed_minus_deepgoplus_control | no_detected_hit | 356 | +0.0854 [+0.0732, +0.0974] |
| CCO | sequence_routed_minus_sequence_static | all | 1278 | +0.0074 [+0.0035, +0.0117] |
| CCO | sequence_routed_minus_sequence_static | no_close_hit | 887 | -0.0012 [-0.0051, +0.0026] |
| CCO | sequence_routed_minus_sequence_static | close_hit | 391 | +0.0276 [+0.0191, +0.0363] |
| CCO | sequence_routed_minus_sequence_static | no_detected_hit | 356 | +0.0035 [+0.0017, +0.0053] |
| CCO | structure_routed_minus_sequence_routed | all | 1278 | -0.0001 [-0.0005, +0.0003] |
| CCO | structure_routed_minus_sequence_routed | no_close_hit | 887 | -0.0003 [-0.0008, +0.0002] |
| CCO | structure_routed_minus_sequence_routed | close_hit | 391 | +0.0003 [-0.0003, +0.0010] |
| CCO | structure_routed_minus_sequence_routed | no_detected_hit | 356 | +0.0004 [-0.0005, +0.0012] |
| CCO | confidence_routed_minus_structure_routed | all | 1278 | +0.0008 [+0.0002, +0.0013] |
| CCO | confidence_routed_minus_structure_routed | no_close_hit | 887 | +0.0011 [+0.0004, +0.0018] |
| CCO | confidence_routed_minus_structure_routed | close_hit | 391 | +0.0002 [-0.0005, +0.0010] |
| CCO | confidence_routed_minus_structure_routed | no_detected_hit | 356 | +0.0022 [+0.0011, +0.0035] |
| CCO | confidence_routed_minus_shuffled_routed | all | 1278 | +0.0005 [+0.0000, +0.0009] |
| CCO | confidence_routed_minus_shuffled_routed | no_close_hit | 887 | +0.0005 [-0.0001, +0.0010] |
| CCO | confidence_routed_minus_shuffled_routed | close_hit | 391 | +0.0005 [-0.0002, +0.0013] |
| CCO | confidence_routed_minus_shuffled_routed | no_detected_hit | 356 | +0.0005 [+0.0001, +0.0008] |
| CCO | structure_routed_minus_deepgoplus_control | all | 1278 | +0.0199 [+0.0151, +0.0245] |
| CCO | structure_routed_minus_deepgoplus_control | no_close_hit | 887 | +0.0197 [+0.0133, +0.0267] |
| CCO | structure_routed_minus_deepgoplus_control | close_hit | 391 | +0.0274 [+0.0201, +0.0346] |
| CCO | structure_routed_minus_deepgoplus_control | no_detected_hit | 356 | +0.0858 [+0.0734, +0.0981] |
| MFO | sequence_routed_minus_deepgoplus_control | all | 1085 | +0.0199 [+0.0147, +0.0251] |
| MFO | sequence_routed_minus_deepgoplus_control | no_close_hit | 715 | +0.0310 [+0.0227, +0.0387] |
| MFO | sequence_routed_minus_deepgoplus_control | close_hit | 370 | +0.0089 [+0.0051, +0.0127] |
| MFO | sequence_routed_minus_deepgoplus_control | no_detected_hit | 246 | +0.1975 [+0.1752, +0.2193] |
| MFO | sequence_routed_minus_sequence_static | all | 1085 | +0.0100 [+0.0058, +0.0136] |
| MFO | sequence_routed_minus_sequence_static | no_close_hit | 715 | +0.0103 [+0.0046, +0.0155] |
| MFO | sequence_routed_minus_sequence_static | close_hit | 370 | +0.0151 [+0.0105, +0.0201] |
| MFO | sequence_routed_minus_sequence_static | no_detected_hit | 246 | +0.0913 [+0.0772, +0.1064] |
| MFO | structure_routed_minus_sequence_routed | all | 1085 | +0.0002 [-0.0001, +0.0006] |
| MFO | structure_routed_minus_sequence_routed | no_close_hit | 715 | +0.0001 [-0.0003, +0.0006] |
| MFO | structure_routed_minus_sequence_routed | close_hit | 370 | +0.0004 [+0.0000, +0.0009] |
| MFO | structure_routed_minus_sequence_routed | no_detected_hit | 246 | +0.0011 [+0.0001, +0.0023] |
| MFO | confidence_routed_minus_structure_routed | all | 1085 | +0.0002 [-0.0003, +0.0007] |
| MFO | confidence_routed_minus_structure_routed | no_close_hit | 715 | +0.0003 [-0.0004, +0.0010] |
| MFO | confidence_routed_minus_structure_routed | close_hit | 370 | +0.0002 [-0.0005, +0.0008] |
| MFO | confidence_routed_minus_structure_routed | no_detected_hit | 246 | +0.0016 [+0.0002, +0.0033] |
| MFO | confidence_routed_minus_shuffled_routed | all | 1085 | +0.0003 [+0.0000, +0.0007] |
| MFO | confidence_routed_minus_shuffled_routed | no_close_hit | 715 | +0.0006 [+0.0001, +0.0012] |
| MFO | confidence_routed_minus_shuffled_routed | close_hit | 370 | -0.0001 [-0.0004, +0.0002] |
| MFO | confidence_routed_minus_shuffled_routed | no_detected_hit | 246 | +0.0009 [-0.0000, +0.0020] |
| MFO | structure_routed_minus_deepgoplus_control | all | 1085 | +0.0201 [+0.0149, +0.0254] |
| MFO | structure_routed_minus_deepgoplus_control | no_close_hit | 715 | +0.0312 [+0.0226, +0.0389] |
| MFO | structure_routed_minus_deepgoplus_control | close_hit | 370 | +0.0094 [+0.0056, +0.0133] |
| MFO | structure_routed_minus_deepgoplus_control | no_detected_hit | 246 | +0.1986 [+0.1765, +0.2202] |
