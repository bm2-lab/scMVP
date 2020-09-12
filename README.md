# scMVP - single cell multi-view processor

scMVP is a python toolkit for joint profiling of scRNA and scATAC data profiling and analysis
with multi-modal VAE model.

## Installation
**Environment requirements:**<br>
scMVP requires Python3.7.x and [**Pytorch**](http://pytorch.org).<br>
For example, use [**miniconda**](https://conda.io/miniconda.html) to install python and pytorch of CPU or GPU version.
```Bash
conda install -c pytorch python=3.7 pytorch
# if you do not have jupyter notebook/ipython notebook, you can also install by conda
conda install jupyter
```

Then you can install scMVP from github repo:<br>
```Bash
# first move to your target directory
git clone https://github.com/bm2-lab/scMVP.git
cd scMVP/
python setup.py install
```

Try ```import scMVP``` in your python console and start your first [**tutorial**](demos/scMVP_tutorial.ipynb) with scMVP!

## Data preparation
Your should first prepare your input files, example is as follows:

1. "XX_cell.tsv": cell barcodes of RNA <br>
2. "XX_gene.count.mtx" or  "XX_gene.count.tsv": gene expression matrix <br>
3. "XX_cDNA.genes.tsv": gene names <br>
4. "XX_cell.ATAC.tsv": cell barcodes of ATAC <br>
5. "XX_chromatin.count.mtx" or  "XX_chromatin.count.tsv": atac expression matrix  <br>
6. "XX_peak.tsv": peak names/ids <br>

**Optional:**<br>
-  "XX_embeddings.xls": given cell annotation labels. <br>

### Bulit in dataset:
- dataset.scienceDataset(): sci-CAR paper dataset.<br>
- dataset.pairedSeqDataset(): Paired-seq paper dataset.<br>
- dataset.snareDataset(): SNARE-seq dataset.<br>


## User tutorial

1. Using scMVP for sci-CAR cell line mixture. [**demo**](demos/scMVP_tutorial.ipynb)
- Basic analysis modules with multi-VAE.

2. Using scMVP for snare-seq mouse cerebral cortex P0 dataset. [**demo**](demos/scMVP_regress_tutorial.ipynb)
- Perform CRE-gene analysis with PLS-regression.

3. Using scMVP on customize joint profiling dataset.[**demo**](demos/scMVP_dataloader.ipynb)
- Load and analyze your own data.


### Reference
scMVP: an integrative generative model for joint profiling of single cell RNA-seq and ATAC-seq data. 2020

