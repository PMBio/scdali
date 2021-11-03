---
layout: default
title: "Usage"
---

<strong>scDALI</strong> requires two inputs: A cell state representation and allele-specific read counts for a set of genomic features and each cell.

## Processing of total counts & cell-state definition

The cell state representation can be generated using established tools for single-cell sequencing analysis from the matrix of total read counts per genomic feature and cell (see for example [here](https://scanpy.readthedocs.io/en/stable/tutorials.html)) The choice of a suitable representation can depend both on the data and research question. However, unless you are specifically interested in discrete (cell-type-specific) or (pseudo-)temporal effects, we recommend using a lower-dimensional embedding such as PCA to define cell states. While allele-specific analysis are generally robust to technical effects, we also recommend that you follow best practices for single-cell sequencing analysis and account for confounding variables such as batch effects or library size when defining the cell-state representation.

## Generation of allele-specific counts

Allele-specific read counts can be generated using heterozygous genetic variants as natural barcodes. We recommend following the approach taken by [WASP](https://github.com/bmvdgeijn/WASP), a suite of tools for the unbiased quantification of allele-specific read counts. In particular, WASP involves a filtering step to reduce potential reference mapping biases. While mapping biases will not lead to false positive results when testing for heterogeneous allelic imbalance (as all cells will be affected equally), they do need to be removed before applying scDALI-Hom (test for homogeneous imbalance) or scDALI-Joint (test for heterogeneous or homogeneous imbalance).

Note that WASP was originally developped for bulk sequencing data. We will update this site to include a reference to a modified version for single-cell data in the future.

## Running the scDALI test

Once a cell-state representation and allele-specific counts have been generated, we can use the high-level interface to run the scDALI tests for multiple genomic features. Shown below is a toy example:

    from scdali import run_tests

    # generate toy data
    n_cells = 100
    n_genes = 200
    cell_state_dimension = 5

    cell_state = np.random.normal(size=(n_cells, cell_state_dimension))  
    
    # sample total counts (maternal + paternal)
    D = np.random.poisson(10, size=(n_cells, n_genes)) 
    
    # sample alternative counts, e.g. maternal
    A = np.random.binomial(p=0.5, n=D) 

    # run scDALI tests
    pvalues = run_tests(A=A, D=D, model='scDALI-Het', cell_state=cell_state, n_cores=1)['pvalues']
    print(np.median(pvalues)) # approximately 0.5 (no cell-state-specific effects)
  
To learn more about the different function arguments, we can call 

    help(run_tests)

## Downstream analysis & interpretation


## Low-level scDALI interface
