#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


requirements = [
    "numpy>=1.16.2",
    "torch>=1.0.1",
    "matplotlib>=3.0.3",
    "h5py>=2.9.0",
    "pandas>=0.24.2",
    "loompy>=2.0.16",
    "tqdm>=4.31.1",
    "xlrd==1.2.0",
    # "nbconvert>=5.4.0",
    # "nbformat>=4.4.0",
    # "jupyter>=1.0.0",
    # "ipython>=7.1.1",
    # "anndata==0.6.22.post1",
    # "scanpy==1.4.4.post1",
    "dask==2.0",
    "anndata>=0.7",
    "scanpy>=1.4.6",
    "scikit-learn==0.22.2",
    "numba>=0.48",  # numba 0.45.1 has a conflict with UMAP and numba 0.46.0 with parallelization in loompy
    "hyperopt==0.1.2",
]

setup_requirements = ["pip>=18.1"]

test_requirements = [
    "pytest>=3.7.4",
    "pytest-runner>=2.11.1",
    "flake8>=3.7.7",
    "coverage>=4.5.1",
    "codecov>=2.0.8",
    "black>=19.3b0",
]

extras_requirements = {
    "notebooks": [
        "louvain>=0.6.1",
        "python-igraph>=0.7.1.post6",
        "colour>=0.1.5",
        "umap-learn>=0.3.8",
        "seaborn>=0.9.0",
        "leidenalg>=0.7.0",
    ],
    "docs": [
        "sphinx>=1.7.1",
        "nbsphinx",
        "sphinx_autodoc_typehints",
        "sphinx-rtd-theme",
    ],
    "test": test_requirements,
}
author = (
    "Gao yang Li, Shaliu FU"
)

setup(
    author=author,
    author_email="lgyzngc@tongji.edu.cn",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: bioinformatics",
    ],
    description="Single Cell Multi-View Profiler",
    install_requires=requirements+extras_requirements["notebooks"],
    license="MIT license",
    # long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="scMVP",
    name="scMVP",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extras_requirements,
    url="https://github.com/bm2-lab/scMVP",
    version="0.0.1",
    zip_safe=False,
)
