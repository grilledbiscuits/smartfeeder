# Field accuracy by framing

2844 of 9139 field frames are portrait video padded into a landscape frame (68% black bars). The deployment camera cannot produce them, so they are reported separately rather than pooled.

*Argmax over the taxon head only: no prior, temperature, novelty gate, rollup, per-class thresholds or clip vote. Framing comparison, not deployment performance.*

### FP32 checkpoint (all splits)

| Split / class | Deployment framing | Pillarboxed |
|---|---|---|
| **test, all field birds** | **0.943 (n=579)** | **0.914 (n=580)** |
| Amethyst Sunbird | 0.876 (n=186) | — |
| Southern Double-collared Sunbird | 0.989 (n=264) | 0.877 (n=400) |
| other_animal | 0.000 (n=6) | — |
| Cape Bulbul | 0.988 (n=80) | — |
| Cape White-eye | 1.000 (n=43) | 0.994 (n=180) |
| **train, all field birds** | **0.997 (n=4550)** | **0.984 (n=1539)** |
| Amethyst Sunbird | 0.987 (n=450) | 0.945 (n=382) |
| Southern Double-collared Sunbird | 0.999 (n=2202) | 0.997 (n=944) |
| Fork-tailed Drongo | 1.000 (n=819) | — |
| other_animal | 1.000 (n=20) | 1.000 (n=8) |
| Cape Bulbul | 1.000 (n=313) | — |
| Cape White-eye | 0.995 (n=746) | 1.000 (n=205) |
| **val, all field birds** | **0.927 (n=690)** | **0.880 (n=725)** |
| Amethyst Sunbird | 0.762 (n=202) | — |
| Southern Double-collared Sunbird | 0.998 (n=485) | 0.892 (n=204) |
| Fork-tailed Drongo | — | 0.927 (n=274) |
| other_animal | — | 0.778 (n=9) |
| Cape Bulbul | — | 0.246 (n=57) |
| Cape White-eye | 0.667 (n=3) | 1.000 (n=181) |

### INT8 graph with capture preprocessing — the deployed path (validation)

| Split / class | Deployment framing | Pillarboxed |
|---|---|---|
| **val, all field birds** | **0.933 (n=690)** | **0.793 (n=725)** |
| Amethyst Sunbird | 0.782 (n=202) | — |
| Southern Double-collared Sunbird | 0.998 (n=485) | 0.760 (n=204) |
| Fork-tailed Drongo | — | 0.854 (n=274) |
| other_animal | — | 0.778 (n=9) |
| Cape Bulbul | — | 0.070 (n=57) |
| Cape White-eye | 0.667 (n=3) | 0.967 (n=181) |
