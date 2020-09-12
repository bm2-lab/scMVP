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


class ATACDataset(GeneExpressionDataset):
    """Loads a file from `10x`_ website.

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
        >>> atac_dataset = ATACDataset(RNA_data,gene_name,cell_name)

    """


    def __init__(
        self,
        ATAC_data: np.matrix = None,
        ATAC_name: pd.DataFrame = None,
        cell_name: pd.DataFrame = None,
        delayed_populating: bool = False,
        is_filter = True,
        datatype="atac_seq",
    ):


        if ATAC_data.all() == None:
            raise Exception("Invalid Input, the gene expression matrix is empty!")
        self.ATAC_data = ATAC_data
        self.ATAC_name = ATAC_name
        self.cell_name = cell_name
        self.is_filter = is_filter
        self.datatype = datatype
        self.cell_name_formulation = None
        self.atac_name_formulation = None

        if not isinstance(self.ATAC_name, pd.DataFrame):
            self.ATAC_name = pd.DataFrame(self.ATAC_name)
        if not isinstance(self.cell_name, pd.DataFrame):
            self.cell_name = pd.DataFrame(self.cell_name)


        # form data name and filename unless manual override
        logger.debug("Loading atac expression dataset")
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing dataset")
        data_dict = {}
        data_dict["cell_names"] = self.cell_name
        if self.is_filter:
            temp = self.ATAC_data
            high_exp_gene = ((temp > 0).sum(axis=1).ravel() >= 3)
            data_dict["atac_expression"] = temp[high_exp_gene,:]
            data_dict["atac_names"] = self.ATAC_name.loc[high_exp_gene, :]
        else:
            data_dict["atac_expression"] = self.ATAC_data
            data_dict["atac_names"] = self.ATAC_name

        cell_attributes_dict = {
            "atac_names": np.squeeze(np.asarray(data_dict["atac_names"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=data_dict["atac_expression"],
            batch_indices=None,
            gene_names=data_dict["cell_names"].astype(np.str),
            cell_attributes_dict=cell_attributes_dict,
        )
        self.filter_cells_by_count(datatype=self.datatype)
        self.cell_name_formulation = np.array(data_dict["cell_names"])
        self.atac_name_formulation = np.array(data_dict["atac_names"])
        '''
        index = np.array([0,1,2,4])
        index_id = np.array(range(len(self.atac_name_formulation)))
        data_test = self.atac_name_formulation[index_id == index[0]]
        data_test = self.atac_name_formulation[index_id == index[0]]
        print(self.atac_name_formulation[index_id == index[0]])
        print(self.atac_name_formulation[index_id == index])
        '''
