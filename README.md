# scMVP - single cell Multi-View Profiler

scMVP is a python toolkit for joint profiling of scRNA and scATAC data profiling and analysis
with multi-modal VAE model.

## Installation
**Environment requirements:**<br>
scMVP requires Python3.7.x and [**Pytorch**](http://pytorch.org).<br>
For example, use [**miniconda**](https://conda.io/miniconda.html) to install python and pytorch of CPU or GPU version.
```Bash
conda install -c pytorch python=3.7 pytorch
# if you do not have jupyter notebook/ipython notebook, you can also install it by conda
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

1. "XX.tsv": cell barcodes of RNA <br>
2. "XX.mtx" or  "XX.tsv": gene expression matrix <br>
3. "XX.tsv": gene names <br>
4. "XX.tsv": cell barcodes of ATAC <br>
5. "XX.mtx" or  "XX.tsv": atac expression matrix  <br>
6. "XX.tsv": peak names/ids <br>

**OR** <br>
<br>
1. "XX.tsv": gene expression dense matrix with rownames(genes) and colnames(barcodes)<br>
2. "XX.tsv": atac expression dense matrix with rownames(genes) and colnames(barcodes)<br>

**Optional:**<br>
- Custom cell annotation labels or other labels. <br>


### Bulit in dataset:
- dataset.SciCarDemo(): sci-CAR paper dataset.<br>
- dataset.PairedDemo(): Paired-seq paper dataset.<br>
- dataset.SnareDemo(): SNARE-seq paper dataset.<br>

### pretrained models:<br>
Download link: [baidu cloud disk](https://pan.baidu.com/s/1_CR1b-GEVM1xeys9ks5Ocw)<br>
Download code: lg5v<br>

## User tutorial

1. Using scMVP for sci-CAR cell line mixture. [**demo**](demos/scMVP_tutorial.ipynb)
- Basic analysis modules with multi-VAE.

2. Load published or new joint profiling dataset to scMVP. [**demo**](demos/scMVP_dataloader.ipynb)
- Load and analyze your own data.


### Reference
A deep generative model for multi-view profiling of single-cell RNA-seq and ATAC-seq data.(submited) 2020

