import logging
import os
import tarfile
import urllib
import numpy as np
import pandas as pd
import scipy.io as sp_io
from scipy.sparse import csr_matrix, issparse

from scMVP.dataset.dataset import CellMeasurement, GeneExpressionDataset

logger = logging.getLogger(__name__)

available_specification = ["filtered", "raw"]


class LoadData(GeneExpressionDataset):
    """
    Dataset format:
    dataset = {
    "gene_barcodes": xxx,
    "gene_expression": xxx,
    "gene_names": xxx,
    "atac_barcodes": xxx,
    "atac_expression": xxx,
    "atac_names": xxx,
    }
    OR
    dataset = {
    "gene_expression":xxx,
    "atac_expression":xxx,
    }
    """

    def __init__(self,
        dataset: dict = None,
        data_path: str = "dataset/",
        dense: bool = False,
        measurement_names_column: int = 0,
        remove_extracted_data: bool = False,
        delayed_populating: bool = False,
        file_separator: str = "\t",
        gzipped: bool = False,
        atac_threshold: float = 0.0001, # express in over 0.01%
        cell_threshold: int = 1 # filtering cells less than minimum count
                 ):

        self.dataset = dataset
        self.data_path = data_path
        self.barcodes = None
        self.dense = dense
        self.measurement_names_column = measurement_names_column
        self.remove_extracted_data = remove_extracted_data
        self.file_separator = file_separator
        self.gzip = gzipped
        self.atac_thres = atac_threshold
        self.cell_thres = cell_threshold
        self._minimum_input = ("gene_expression", "atac_expression")
        self._allow_input = (
                             "gene_expression", "atac_expression",
                             "gene_barcodes", "gene_names",
                             "atac_barcodes", "atac_names"
                             )
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing joint profiling dataset.")
        if not self._input_check():
            logging.debug("Please reload your dataset.")
            return
        joint_profiles = {}
        if len(self.dataset.keys()) == 2:
            for _key in self.dataset.keys():
                if self.gzip:
                    joint_profiles[_key] = pd.read_csv(self.dataset[_key], sep=self.file_separator,
                        header=0, index_col=0)
                else:
                    joint_profiles[_key] = pd.read_csv(self.dataset[_key], sep=self.file_separator,
                                                       header=0, index_col=0, compression="gzip")

        elif len(self.dataset.keys()) == 6:
            for _key in self.dataset.keys():
                if self.gzip:
                    if _key.endswith("mtx.gz"):
                        if self.dense:
                            joint_profiles[_key] = sp_io.mmread(self.dataset[_key]).T
                        else:
                            joint_profiles[_key] = csr_matrix(sp_io.mmread(self.dataset[_key]).T)
                    else:
                        pd.read_csv(self.dataset[_key], sep=self.file_separator,
                                    compression="gzip", header=None)
                else:
                    if _key.endswith("mtx"):
                        if self.dense:
                            joint_profiles[_key] = sp_io.mmread(self.dataset[_key]).T
                        else:
                            joint_profiles[_key] = csr_matrix(sp_io.mmread(self.dataset[_key]).T)
                    else:
                        pd.read_csv(self.dataset[_key], sep=self.file_separator, header=None)

        share_index, gene_barcode_index, atac_barcode_index = np.intersect1d(joint_profiles["gene_barcodes"].values,
                                                                    joint_profiles["atac_barcodes"].values,
                                                                    return_indices=True)

        # reorder rnaseq cell meta
        tmp = joint_profiles["gene_barcodes"]
        joint_profiles["gene_barcodes"] = tmp.loc[gene_barcode_index,:]
        gene_tab = joint_profiles["gene_expression"]
        if issparse(gene_tab):
            joint_profiles["gene_expression"] = gene_tab[gene_barcode_index,:].A
        else:
            joint_profiles["gene_expression"] = gene_tab[gene_barcode_index, :]

        temp = joint_profiles["atac_barcodes"]
        joint_profiles["atac_barcodes"] = temp.loc[atac_barcode_index, :]
        temp = joint_profiles["atac_expression"]
        reorder_atac_exp = temp[atac_barcode_index, :]
        binary_index = reorder_atac_exp > 1
        reorder_atac_exp[binary_index] = 1
        # remove peaks > 10% of total cells
        high_count_atacs = ((reorder_atac_exp > 0).sum(axis=0).ravel() >= self.atac_thres * reorder_atac_exp.shape[0]) \
                           & ((reorder_atac_exp > 0).sum(axis=0).ravel() <= 0.1 * reorder_atac_exp.shape[0])

        if issparse(reorder_atac_exp):
            high_count_atacs_index = np.where(high_count_atacs)
            _tmp = reorder_atac_exp[:, high_count_atacs_index[1]]
            reorder_atac_exp["atac_expression"] = _tmp.A
        else:
            _tmp = reorder_atac_exp[:, high_count_atacs]
            reorder_atac_exp["atac_expression"] = _tmp
            reorder_atac_exp["atac_names"] = reorder_atac_exp["atac_names"].loc[high_count_atacs, :]

         # RNA-seq as the key
        Ys = []
        measurement = CellMeasurement(
            name="atac_expression",
            data=joint_profiles["atac_expression"],
            columns_attr_name="atac_names",
            columns=joint_profiles["atac_names"].astype(np.str),
        )
        Ys.append(measurement)

        cell_attributes_dict = {
            "barcodes": np.squeeze(np.asarray(joint_profiles["gene_barcodes"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=joint_profiles["gene_expression"],
            batch_indices=None,
            gene_names=joint_profiles["gene_names"].astype(np.str),
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
        )
        self.filter_cells_by_count(self.cell_thres)

    def _input_check(self):
        if len(self.dataset.keys()) == 2:
            for _key in self.dataset.keys():
                if _key not in self._minimum_input:
                    logger.debug("Unknown input data type:{}".format(_key))
                    return False
                # if not self.dataset[_key].split(".")[-1] in ["txt","tsv","csv"]:
                #     logger.debug("scMVP only support two files input of txt, tsv or csv!")
                #     return False
        elif len(self.dataset.keys()) >= 6:
            for _key in self._allow_input:
                if not _key in self.dataset.keys():
                    logger.debug("Data type {} missing.".format(_key))
                    return False
        else:
            logger.debug("Incorrect input file number.")
            return False
        for _key in self.dataset.keys():
            if not os.path.exists(self.data_path):
                logger.debug("{} do not exist!".format(self.data_path))
            if not os.path.exists("{}{}".format(self.data_path, self.dataset[_key])):
                logger.debug("Cannot find {}{}!".format(self.data_path, self.dataset[_key]))
                return False
        return True

    def _download(url: str, save_path: str, filename: str):
        """Writes data from url to file."""
        if os.path.exists(os.path.join(save_path, filename)):
            logger.info("File %s already downloaded" % (os.path.join(save_path, filename)))
            return

        r = urllib.request.urlopen(url)
        logger.info("Downloading file at %s" % os.path.join(save_path, filename))

        def read_iter(file, block_size=1000):
            """Given a file 'file', returns an iterator that returns bytes of
            size 'blocksize' from the file, using read()."""
            while True:
                block = file.read(block_size)
                if not block:
                    break
                yield block


class SnareDemo(LoadData):

    def __init__(self, dataset_name: str=None, data_path: str="/dataset"):
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074"
        available_datasets = {
            "CellLineMixture": {
                "gene_expression": "GSE126074_CellLineMixture_SNAREseq_cDNA.counts.tsv.gz",
                "atac_expression": "GSE126074_CellLineMixture_SNAREseq_chromatin.counts.tsv.gz",
            },
            "AdBrainCortex": {
                "gene_barcodes": "GSE126074_AdBrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
                "gene_expression": "GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz",
                "gene_names": "GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv.gz",
                "atac_barcodes": "GSE126074_AdBrainCortex_SNAREseq_chromatin.barcodes.tsv.gz",
                "atac_expression": "GSE126074_AdBrainCortex_SNAREseq_chromatin.counts.mtx.gz",
                "atac_names": "GSE126074_AdBrainCortex_SNAREseq_chromatin.peaks.tsv.gz",
            },
            "P0_BrainCortex": {
                "gene_barcodes": "GSE126074_P0_BrainCortex_SNAREseq_cDNA.barcodes.tsv.gz",
                "gene_expression": "GSE126074_P0_BrainCortex_SNAREseq_cDNA.counts.mtx.gz",
                "gene_names": "GSE126074_P0_BrainCortex_SNAREseq_cDNA.genes.tsv.gz",
                "atac_barcodes": "GSE126074_P0_BrainCortex_SNAREseq_chromatin.barcodes.tsv.gz",
                "atac_expression": "GSE126074_P0_BrainCortex_SNAREseq_chromatin.counts.mtx.gz",
                "atac_names": "GSE126074_P0_BrainCortex_SNAREseq_chromatin.peaks.tsv.gz",
            }
        }
        if dataset_name=="CellLineMixture":
            super().__init__(dataset = available_datasets[dataset_name],
                         data_path= data_path,
                         dense = False,
                         measurement_names_column = 0,
                         remove_extracted_data = False,
                         delayed_populating = False,
                         file_separator = "\t",
                         gzipped = True,
                         atac_threshold = 0.0005,
                         cell_threshold = 1
                         )
        elif dataset_name=="AdBrainCortex" or dataset_name=="P0_BrainCortex":
            super().__init__(dataset=available_datasets[dataset_name],
                             data_path=data_path,
                             dense=True,
                             measurement_names_column=0,
                             remove_extracted_data=False,
                             delayed_populating=False,
                             gzipped=True,
                             atac_threshold=0.0005,
                             cell_threshold=1
                             )
        else:
            logger.info('Please select from "CellLineMixture", "AdBrainCortex" or "P0_BrainCortex" dataset.')


class PariedDemo(LoadData):

    def __init__(self, dataset_name: str = None, data_path: str = "/dataset"):
        urls = [
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130399/suppl/GSE130399_GSM3737488_GSM3737489_Cell_Mix.tar.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130399/suppl/GSE130399_GSM3737490-GSM3737495_Adult_Cerebrail_Cortex.tar.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130399/suppl/GSE130399_GSM3737496-GSM3737499_Fetal_Forebrain.tar.gz"
                ]

        available_datasets = {
            "CellLineMixture": {
                "gene_barcodes": "Cell_Mix_RNA/genes.tsv",
                "gene_expression": "Cell_Mix_RNA/matrix.mtx",
                "gene_names": "Cell_Mix_RNA/barcodes.tsv",
                "atac_barcodes": "Cell_Mix_DNA/genes.tsv",
                "atac_expression": "Cell_Mix_DNA/matrix.mtx",
                "atac_names": "Cell_Mix_DNA/barcodes.tsv"
            },
            "Adult_Cerebral": {
                "gene_barcodes": "Adult_CTX_RNA/genes.tsv",
                "gene_expression": "Adult_CTX_RNA/matrix.mtx",
                "gene_names": "Adult_CTX_RNA/barcodes.tsv",
                "atac_barcodes": "Adult_CTX_DNA/genes.tsv",
                "atac_expression": "Adult_CTX_DNA/matrix.mtx",
                "atac_names": "Adult_CTX_DNA/barcodes.tsv"
            },
            "Fetal_Forebrain": {
                "gene_barcodes": "FB_RNA/genes.tsv",
                "gene_expression": "FB_RNA/matrix.mtx",
                "gene_names": "FB_RNA/barcodes.tsv",
                "atac_barcodes": "FB_DNA/genes.tsv",
                "atac_expression": "FB_DNA/matrix.mtx",
                "atac_names": "FB_DNA/barcodes.tsv"
            }
        }
        if dataset_name=="CellLineMixture":
            super().__init__(dataset = available_datasets[dataset_name],
                         data_path= data_path,
                         dense = False,
                         measurement_names_column = 0,
                         remove_extracted_data = False,
                         delayed_populating = False,
                         gzipped = False,
                         atac_threshold = 0.0005,
                         cell_threshold = 1
                         )
        elif dataset_name=="Adult_Cerebral" or dataset_name=="Fetal_Forebrain":
            super().__init__(dataset=available_datasets[dataset_name],
                             data_path=data_path,
                             dense=True,
                             measurement_names_column=0,
                             remove_extracted_data=False,
                             delayed_populating=False,
                             gzipped=True,
                             atac_threshold=0.0005,
                             cell_threshold=1
                             )
        else:
            logging.info('Please select from {} dataset.'.format("\t".join(available_datasets.keys())))


class SciCarDemo(LoadData):
    def __init__(self, dataset_name: str = None, data_path: str = "/dataset"):
        urls = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE117089&format=file"
        # NOTICE, tsv files are generated from original txt files
        available_datasets = {
            "CellLineMixture": {
                "gene_barcodes": "GSM3271040_RNA_sciCAR_A549_cell.tsv",
                "gene_names": "GSM3271040_RNA_sciCAR_A549_gene.tsv",
                "gene_expression": "GSM3271040_RNA_sciCAR_A549_gene.count.mtx",
                "atac_barcodes": "GSM3271041_ATAC_sciCAR_A549_cell.tsv",
                "atac_names": "GSM3271041_ATAC_sciCAR_A549_peak.tsv",
                "atac_expression": "GSM3271041_ATAC_sciCAR_A549_peak.count.mtx"
            },
            "mouse_kidney": {
                "gene_barcodes": "GSM3271044_RNA_mouse_kidney_cell.tsv",
                "gene_names": "GSM3271044_RNA_mouse_kidney_gene.tsv",
                "gene_expression": "GSM3271044_RNA_mouse_kidney_gene.count.mtx",
                "atac_barcodes": "GSM3271045_ATAC_mouse_kidney_cell.tsv",
                "atac_names": "GSM3271045_ATAC_mouse_kidney_peak.tsv",
                "atac_expression": "GSM3271045_ATAC_mouse_kidney_peak_count.mtx"
            }
        }
        if dataset_name:
            for barcode_file in ["gene_barcodes", "atac_barcodes", "gene_names", "atac_names"]:
                # generate gene and atac barcodes from cell metadata.
                with open("{}/{}".format(data_path, available_datasets[dataset_name][barcode_file]),"w") as fo:
                    infile = "{}/{}.txt".format(data_path, available_datasets[dataset_name][barcode_file][:-4])
                    indata = [i.rstrip().split(",") for i in infile][1:]
                    for line in indata:
                        fo.write("{}\n".format(line[0]))

            super().__init__(dataset=available_datasets[dataset_name],
                             data_path=data_path,
                             dense=True,
                             measurement_names_column=0,
                             remove_extracted_data=False,
                             delayed_populating=False,
                             gzipped=False,
                             atac_threshold=0.0005,
                             cell_threshold=1
                             )
        else:
            logging.info('Please select from {} dataset.'.format("\t".join(available_datasets.keys())))


# obselete demo
class scMVP_dataloader(GeneExpressionDataset):
    """Loads a file from user.
    :param dataset_name: Name of the dataset file. Has to be a dict included 6 or 2 elements
         with the format of: {"XX_cDNA.barcodes.tsv":"gene_barcodes",
         "XX_cDNA.counts.mtx": "gene_expression",
         "XX_cDNA.genes.tsv": "gene_names",
        "XX_chromatin.barcodes.tsv":"atac_barcodes",
        "XX_chromatin.counts.mtx":"atac_expression",
        "XX_chromatin.peaks.tsv":"atac_names"}
        or {"XX_cDNA.counts.tsv": "gene_expression",
    "XX_chromatin.counts.tsv":"atac_expression",}.
    where XX represent the specific file prefix,
    such as: GSE126074_AdBrainCortex_SNAREseq
    :param save_path: Location to use when saving/loading the data.
    :param type: Either `filtered` data or `raw` data.
    :param dense: Whether to load as dense or sparse.
        If False, data is cast to sparse using ``scipy.sparse.csr_matrix``.
    :param measurement_names_column: column in which to find measurement names in the corresponding `.tsv` file.
    :param remove_extracted_data: Whether to remove extracted archives after populating the dataset.
    :param delayed_populating: Whether to populate dataset with a delay
    :param is_binary: is binary of the input atac data
    Examples:
        >>> my_dataset = scMVP_dataloader(\
        {"GSE126074_AdBrainCortex_SNAREseq_cDNA.barcodes.tsv":"gene_barcodes",\
        "GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx": "gene_expression",\
        "GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv": "gene_names",\
        "GSE126074_AdBrainCortex_SNAREseq_chromatin.barcodes.tsv":"atac_barcodes",\
        "GSE126074_AdBrainCortex_SNAREseq_chromatin.counts.mtx":"atac_expression",\
        "GSE126074_AdBrainCortex_SNAREseq_chromatin.peaks.tsv":"atac_names"}\
        )
    """


    def __init__(
        self,
        dataset_name: dict = None,
        save_path: str = "dataset/",
        type: str = "filtered",
        dense: bool = False,
        measurement_names_column: int = 0,
        remove_extracted_data: bool = False,
        delayed_populating: bool = False,
        datatype = "atac_seq",
        is_binary = False,
    ):

        self.dataname = dataset_name
        self.barcodes = None
        self.dense = dense
        self.measurement_names_column = measurement_names_column
        self.remove_extracted_data = remove_extracted_data
        self.datatype = datatype
        self.is_binary = is_binary

        # form data name and filename unless manual override
        self.save_path_list = []
        if dataset_name is not None:
            filenames = dataset_name.keys()
            for filename in filenames:
                self.save_path_list.append(os.path.join(save_path, filename))
        else:
            logger.debug("Loading extracted user dataset with custom filename")
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing dataset")

        was_extracted = False
        if len(self.save_path_list) > 0:
            for file_path in self.save_path_list:
                if tarfile.is_tarfile(file_path):
                    logger.info("Extracting tar file")
                    tar = tarfile.open(file_path, "r:gz")
                    tar.extractall(path=self.save_path)
                    was_extracted = True
                    tar.close()

        data_dict = {}
        for file_path in self.save_path_list:
            if file_path.split("_")[-1].split(".")[-1] == "mtx" and self.dense:
                data_dict[self.dataname[file_path.split("/")[-1]]] = sp_io.mmread(file_path).T
            elif not self.dense and file_path.split("_")[-1].split(".")[-1] == "mtx":
                data_dict[self.dataname[file_path.split("/")[-1]]] = csr_matrix(sp_io.mmread(file_path).T)
            else:
                if len(self.save_path_list) == 2:
                    data_dict[self.dataname[file_path.split("/")[-1]]] = pd.read_csv(file_path, sep="\t",
                                                                                        header=0, index_col=0)
                else:
                    data_dict[self.dataname[file_path.split("/")[-1]]] = pd.read_csv(file_path, sep="\t", header=None)

        if len(self.save_path_list) == 2:
            temp = data_dict["gene_expression"]
            data_dict["gene_barcodes"] = pd.DataFrame(temp.columns.values)
            data_dict["gene_names"] = pd.DataFrame(temp._stat_axis.values)
            data_dict["gene_expression"] = np.array(temp).T

            temp = data_dict["atac_expression"]
            data_dict["atac_barcodes"] = pd.DataFrame(temp.columns.values)
            data_dict["atac_names"] = pd.DataFrame(temp._stat_axis.values)
            data_dict["atac_expression"] = np.array(temp).T



        #gene_barcode_index = np.array(data_dict["gene_barcodes"]).argsort()
        #gene_barcode_index = data_dict["gene_barcodes"].sort_values(by = ["0"],axis = 0).index.tolist()
        gene_barcode_index = data_dict["gene_barcodes"].values.tolist()
        gene_barcode_index = sorted(range(len(gene_barcode_index)),key = lambda k:gene_barcode_index[k])
        #gene_barcode_index = data_dict["gene_barcodes"].values.tolist().index(gene_barcode_index)
        temp = data_dict["gene_barcodes"]
        data_dict["gene_barcodes"] = temp.loc[gene_barcode_index,:]
        temp = data_dict["gene_expression"]
        if issparse(temp):
            data_dict["gene_expression"] = temp[gene_barcode_index,:].A
        else:
            data_dict["gene_expression"] = temp[gene_barcode_index, :]


        atac_barcode_index = data_dict["atac_barcodes"].values.tolist()
        atac_barcode_index = sorted(range(len(atac_barcode_index)), key=lambda k: atac_barcode_index[k])
        temp = data_dict["atac_barcodes"]
        data_dict["atac_barcodes"] = temp.loc[atac_barcode_index,:]
        temp = data_dict["atac_expression"]
        data_dict["atac_expression"] = temp[atac_barcode_index, :]

        #if issparse(temp):
        #    data_dict["atac_expression"] = temp[atac_barcode_index, :].A
        #else:
        #    data_dict["atac_expression"] = temp[atac_barcode_index, :]
        # filter the atac data
        temp = data_dict["atac_expression"]
        # for binary distribution
        if self.is_binary:
            temp_index = temp > 1
            temp[temp_index] = 1
        # end binary
        high_count_atacs = ((temp > 0).sum(axis=0).ravel() >= 0.005 * temp.shape[0])\
                            & ((temp > 0).sum(axis=0).ravel() <= 0.1 * temp.shape[0])
        #high_count_atacs = ((temp > 0).sum(axis=0).ravel() >= 0.19 * temp.shape[0])



        if issparse(temp):
            high_count_atacs_index = np.where(high_count_atacs)
            temp = temp[:, high_count_atacs_index[1]]
            data_dict["atac_expression"] = temp.A
            data_dict["atac_names"] = data_dict["atac_names"].loc[high_count_atacs_index[1], :]
        else:
            temp = temp[:, high_count_atacs]
            data_dict["atac_expression"] = temp
            data_dict["atac_names"] = data_dict["atac_names"].loc[high_count_atacs, :]
            print(len(temp[temp > 1]))
            print(len(temp[temp < 0]))

         # RNA-seq as the key
        Ys = []
        measurement = CellMeasurement(
            name="atac_expression",
            data=data_dict["atac_expression"],
            columns_attr_name="atac_names",
            columns=data_dict["atac_names"].astype(np.str),
        )
        Ys.append(measurement)

        cell_attributes_dict = {
            "barcodes": np.squeeze(np.asarray(data_dict["gene_barcodes"], dtype=str))
        }

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=data_dict["gene_expression"],
            batch_indices=None,
            gene_names=data_dict["gene_names"].astype(np.str),
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
        )
        self.filter_cells_by_count()

