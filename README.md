# scDALI - Discovering allelic imbalance in single-cells

**scDALI** (single-cell differential allelic imbalance) is a statistical framework for detecting allelic imbalance from
single-cell sequencing data [1].

## Install scDALI
To install the latest version from GitHub, you can use `pip`:
```bash
pip install git+https://github.com/PMBio/scdali
```

## Overview
**scDALI** is intended for the application to single-cell sequencing data such as transcriptome (scRNA-seq) or open-chromatin (e.g. scATAC-seq) measurements. At the core is a Beta-Binomial generalized linear mixed-effects model, capturing allelic ratios as a function of the cell state while accounting for Binomial variance and residual overdispersion. Here, *cell states* are usually defined based on the total (non-allele-specific) signal. Depending on the application, a suitable cell-state definition could be

- the position along a (pseudo-) temporal trajectory
- a cell clustering / cell type annotation
- coordinates in a lower dimensional-embedding, for example, as obtained from PCA, UMAP or Variational autoencoder models (e.g. [SCVI](https://scvi-tools.org/) or our [ATAC-seq specific implementation](https://github.com/tohein/tempo)[1])

scDALI implements three different score-based tests:
- scDALI-Het - test for heterogeneous (cell-state-specific) allelic imbalance
- scDALI-Hom - test for homogeneous allelic imbalance
- scDALI-Joint - test for either kind of allelic imbalance

The scDALI package allows for estimating allelic rates in each cell as a function of the cell state. These estimates can be used for downstream analyses,
such as visualization of variable regions and effect size determination.

## Examples
Check out the example [Jupyter notebook](https://github.com/PMBio/scdali/blob/main/examples/scdali_example.ipynb), highlighting key features.

## Background
Allelic imbalance can be a proxy for *genetic effects*. Significantly variable genes or regions as identified by **scDALI** can be indicative of cell-state-genotype interactions. In addition to differential *total* gene expression or chromatin accessibility, heterogeneous allelic imbalance therefore provides an orthgonal view of cell-state-specific regulation.

![dali abstract](./doc/dali_abstract.png)

## References

[1] T. Heinen, S. Secchia, J. Reddington, B. Zhao, E.E.M. Furlong, O. Stegle. [scDALI: Modelling allelic heterogeneity of DNA accessibility in single-cells reveals context-specific genetic regulation](https://www.biorxiv.org/content/10.1101/2021.03.19.436142v1). Preprint, bioRxiv (2021).
