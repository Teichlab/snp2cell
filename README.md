# snp2cell

snp2cell is a package for identifying gene regulation involved in specific traits and cell types.
It combines three elements: (i) GWAS summary statistics, (ii) single cell data and (iii) a base gene regulatory network.
A network propagation approach is used to integrate and overlap different types of scores on the network.
Random permutations of scores are used to evaluate the significance of high scores. 
A networkx graph of the gene regulatory network with integrated scores can be used to inspect gene regulatory programs that are linked to the trait (from GWAS) on a per cell type basis.

## Installation

```commandline
mamba create -n snp2cell python<3.12
mamba activate snp2cell


```