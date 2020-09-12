import logging
import os
import pickle
import tarfile
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sp_io
import shutil
from scipy.sparse import csr_matrix, issparse

from scMVP.dataset.dataset import CellMeasurement, GeneExpressionDataset, _download

logger = logging.getLogger(__name__)


class geneDataset(GeneExpressionDataset):
    """

    :param dataset_name: Name of the dataset file. Has to be one of:
        "CellLineMixture", "AdBrainCortex", "P0_BrainCortex".
    :param save_path: Location to use when saving/loading the data.
    :param type: Either `filtered` data or `raw` data.
    :param dense: Whether to load as dense or sparse.
        If False, data is cast to sparse using ``scipy.sparse.csr_matrix``.
    :param measurement_names_column: column in which to find measurement names in the corresponding `.tsv` file.
    :param remove_extracted_data: Whether to remove extracted archives after populating the dataset.
    :param delayed_populating: Whether to populate dataset with a delay

    Examples:
        >>> gene_dataset = geneDataset(RNA_data,gene_name,cell_name)

    """


    def __init__(
        self,
        RNA_data: np.matrix = None,
        gene_name: pd.DataFrame = None,
        cell_name: pd.DataFrame = None,
        delayed_populating: bool = False,
        is_filter = True,
        datatype="RNA_seq",
    ):


        if RNA_data.all() == None:
            raise Exception("Invalid Input, the gene expression matrix is empty!")
        self.RNA_data = RNA_data
        self.gene_name = gene_name
        self.cell_name = cell_name
        self.is_filter = is_filter
        self.datatype = datatype
        self.cell_name_formulation = None
        self.gene_name_formulation = None

        if not isinstance(self.gene_name, pd.DataFrame):
            self.gene_name = pd.DataFrame(self.gene_name)
        if not isinstance(self.cell_name, pd.DataFrame):
            self.cell_name = pd.DataFrame(self.cell_name)


        # form data name and filename unless manual override
        logger.debug("Loading gene expression dataset")
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing dataset")
        data_dict = {}
        data_dict["cell_names"] = self.cell_name
        if self.is_filter:
            temp = self.RNA_data
            high_exp_gene = ((temp > 0).sum(axis=1).ravel() >= 20)
            data_dict["gene_expression"] = temp[high_exp_gene,:]
            data_dict["gene_names"] = self.gene_name.loc[high_exp_gene, :]
        else:
            data_dict["gene_expression"] = self.RNA_data
            data_dict["gene_names"] = self.gene_name

        cell_attributes_dict = {
            "gene__names": np.squeeze(np.asarray(data_dict["gene_names"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=data_dict["gene_expression"],
            batch_indices=None,
            gene_names=data_dict["cell_names"].astype(np.str),
            cell_attributes_dict=cell_attributes_dict,
        )
        self.filter_cells_by_count(datatype=self.datatype)
        self.cell_name_formulation = np.array(data_dict["cell_names"])
        self.gene_name_formulation = np.array(data_dict["gene_names"])


