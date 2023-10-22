# scMVP - single cell Multi-View Profiler

scMVP is a python toolkit for joint profiling of scRNA and scATAC data and analysis
with multi-modal self-attention generation model.

## Update logs
- 20220815 <br>
Complete the test for GPU version on NVIDA 3090 platform with CUDA version 11.2. <br>
Add  scRNA and scATAC input check in tutorial. <br>


## Installation
**Environment requirements:**<br>
scMVP requires Python3.7.x and [**Pytorch**](http://pytorch.org).<br>
For example, use [**miniconda**](https://conda.io/miniconda.html) to install python and pytorch of CPU or GPU version.  
We have tested the GPU version on NVIDA 1080Ti platform with CUDA version 10.2.  

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

Jupyter notebooks for other datasets analyzed and benchmarked in our GB paper are deposited in [**folder**](demos/manuscript_analysis/)

### All processed dataset and trained models:<br>
Download link: [baidu cloud disk](https://pan.baidu.com/s/183jLROAUuNfVKCeBY4B4DQ)<br>
- update 23/10/22 : google drive link for cellline datasets: [google drive](https://drive.google.com/drive/folders/18ymTLyMb_wD20O4Z2qkOXBQt5yoDTvea?usp=sharing)<br>

Download code: mkij<br>
- pre_trainer.pkl  scRNA pretraining models <br>
- pre_atac_trainer.pkl scATAC pretraining models <br>
- multi_vae_trainer.pkl scMVP training models <br>


## User tutorial

Applying scMVP to sci-CAR cell line mixture. [**demo**](demos/scMVP_tutorial.ipynb)
- Training and visualization with scMVP.
- Pretraining and transferring to scMVP(perform better in large dataset).



### Reference
A deep generative model for multi-view profiling of single-cell RNA-seq and ATAC-seq data. Genome Biology 2022 [**paper**](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02595-6) 


### Contact Authors
Prof. Qi Liu: [qiliu@tongji.edu.cn](qiliu@tongji.edu.cn)<br>
Dr. Gaoyang Li: [lgyzngc@gmail.com](lgyzngc@gmail.com)<br>
Shaliu Fu: [adam.tongji@gmail.com](adam.tongji@gmail.com)<br>

