---
layout: default
title: "FAQ"
---

## Frequently asked questions

- How do I choose a suitable cell-state representation?

  A good representation depends both on the data and the research question. If you are above all interested in discrete effects, using a [one-hot encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html) of your cell clusters will give the highest power. Similarly, a (pseudo-)temporal ordering can be used if time-specific effects are of particular interest. As a general recommendation, a lower-dimensional embedding of the total counts matrix (PCA, factor analysis, autoencoder, ...) will allow to capture both continuous (e.g. differentiation., cell cyle ... ) and discrete (cell type) effects.

- Do I need to include technical confounders in the scDALI model?

  The analysis of allelic variation may be less affected by technical factors and batch than total expression levels. In particular, if both alleles are thought to be affected similarly by technical effects, these cancel out when quantifying allelic rates. In general we advise to correct for technical confounders (batch effects, size factors) in the cell-state representation (rather than incorporating them as fixed effects in the scDALI model). However, as a special case, we recommend including the donor ID as a fixed effect (using a [one-hot encoding](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)) when modeling population-scale data to avoid potential false positives due to donor-specific reference mapping biases. 

- Can I discover new QTL variants with scDALI?

  No, scDALI does not map regulatory variants associated with allelic imbalance. This is because scDALI was developed as a tool for analyzing data from small samples (i.e. individuals), where there is usually insufficient genetic variation to identify causal variants. However, similar to principles from the analysis of genotype-environment interactions (GxE), one could employ a two-step procedure where (1) discovery is focused on the G component only, using established tools for bulk-eQTL discovery and (2) scDALI is applied in a second step to map cell-state specific effects as described in our paper. For an alternative strategy using total (non-allele-specific) signals, have a look at our sister project [CellRegMap](https://limix.github.io/CellRegMap/)!

  

