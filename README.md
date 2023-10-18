![snp2cell](https://github.com/Teichlab/snp2cell/actions/workflows/tests.yml/badge.svg)

# snp2cell

`snp2cell` is a package for identifying gene regulation involved in specific traits and cell types.
It combines three elements: (i) GWAS summary statistics, (ii) single cell data and (iii) a base gene regulatory network.

A network propagation approach is used to integrate and overlap different types of scores on the network.
Random permutations of scores are then used to evaluate the significance of high scores. 

As an output, a networkx graph of the gene regulatory network with integrated scores can be used to inspect gene regulatory programs that are linked to the trait (from GWAS) on a per cell type basis.

<img src="docs/assets/snp2cell_schematic.png" width="300">

## System requirements

<details>
<summary><b>show requirements</b></summary>

### Hardware requirements

`snp2cell` can run on a standard computer with enough RAM to hold the used datasets in memory.
It can make use of multiple CPUs to speed up computations.

### Software requirements

**OS requirements**

The package has been tested on:

- macOS Monterey (12.6.7)
- Linux: Ubuntu 22.04 jammy

**Python requirements**

A python version `>=3.5` and `<3.12` is required for all dependencies to work. 
Various python libraries are used, listed in `setup.py`, including the python scientific stack, `networkx` and `scanpy`.
`snp2cell` and all dependencies can be installed via `pip` (see below).

</details>

## Installation

*Optional: create and activate a new conda environment (with python<3.12):*
```bash
mamba create -n snp2cell "python<3.12"
mamba activate snp2cell
```

**from github**

```bash
pip install git+ssh://git@github.com/Teichlab/snp2cell.git
```

*(installation time: around 2 min)*

## Usage

**Python module**

snp2cell can be imported as a python module (see [notebooks](#example-notebooks) for examples).

Demo: A minimal demo can be found [here](https://github.com/Teichlab/snp2cell/blob/main/docs/source/toy_example.ipynb) as a jupyter notebook and as a unit test in `test/test_toy_example.py`. (*running time: around 12 sec*)

**CLI**

Importing `snp2cell` as a python module gives most flexibility.
Additionally, there is a command line interface. To see all options, run:

```bash
snp2cell --help
```

Optionally, activate autocompletion for the command line tool.
E.g. for bash run:
```bash
snp2cell --install-completion bash
. ~/.bashrc
```

## Example notebooks

- [toy example](https://github.com/Teichlab/snp2cell/blob/main/docs/source/toy_example.ipynb)

## Citation

`snp2cell` is part of the forthcoming manuscript *"A multiomic atlas of human early skeletal development"* by To, Fei, Pett et al. Stay tuned for details!
