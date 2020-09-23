import logging
import os
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
        cell_threshold: int = 1, # filtering cells less than minimum count
        cell_meta: pd.DataFrame = None,
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
        self.cell_meta = cell_meta
        super().__init__()
        if not delayed_populating:
            self.populate()

    def populate(self):
        logger.info("Preprocessing joint profiling dataset.")
        if not self._input_check():
            logger.info("Please reload your dataset.")
            return
        joint_profiles = {}
        if len(self.dataset.keys()) == 2:
            # for _key in self.dataset.keys():
            if self.gzip:
                _tmp = pd.read_csv("{}/{}".format(self.data_path,self.dataset["gene_expression"]), sep=self.file_separator,
                    header=0, index_col=0)
            else:
                _tmp = pd.read_csv("{}/{}".format(self.data_path,self.dataset["gene_expression"]), sep=self.file_separator,
                                                   header=0, index_col=0, compression="gzip")
            joint_profiles["gene_barcodes"] = pd.DataFrame(_tmp.columns.values)
            joint_profiles["gene_names"] = pd.DataFrame(_tmp._stat_axis.values)
            joint_profiles["gene_expression"] = np.array(_tmp).T

            if self.gzip:
                _tmp = pd.read_csv("{}/{}".format(self.data_path,self.dataset["atac_expression"]), sep=self.file_separator,
                    header=0, index_col=0)
            else:
                _tmp = pd.read_csv("{}/{}".format(self.data_path,self.dataset["atac_expression"]), sep=self.file_separator,
                                                   header=0, index_col=0, compression="gzip")
            joint_profiles["atac_barcodes"] = pd.DataFrame(_tmp.columns.values)
            joint_profiles["atac_names"] = pd.DataFrame(_tmp._stat_axis.values)
            joint_profiles["atac_expression"] = np.array(_tmp).T

        elif len(self.dataset.keys()) == 6:
            for _key in self.dataset.keys():
                if _key == "atac_expression" or _key == "gene_expression" and not self.dense:
                    joint_profiles[_key] = csr_matrix(sp_io.mmread("{}/{}".format(self.data_path,self.dataset[_key])).T)

                elif self.gzip:
                    joint_profiles[_key] = pd.read_csv("{}/{}".format(self.data_path, self.dataset[_key]),
                                                           sep=self.file_separator,
                                                           compression="gzip", header=None)
                else:
                    joint_profiles[_key] = pd.read_csv("{}/{}".format(self.data_path,self.dataset[_key]),
                                                       sep=self.file_separator, header=None)

        else:
            logger.info("more than 6 inputs.")

        ## 200920 gene barcode file may include more than 1 column
        if joint_profiles["gene_names"].shape[1] > 1:
            joint_profiles["gene_names"] = pd.DataFrame(joint_profiles["gene_names"].iloc[:,1])
        if joint_profiles["atac_names"].shape[1] > 1:
            joint_profiles["atac_names"] = pd.DataFrame(joint_profiles["atac_names"].iloc[:,1])
        share_index, gene_barcode_index, atac_barcode_index = np.intersect1d(joint_profiles["gene_barcodes"].values,
                                                                    joint_profiles["atac_barcodes"].values,
                                                                    return_indices=True)
        if isinstance(self.cell_meta,pd.DataFrame):
            if self.cell_meta.shape[1] < 2:
                logger.info("Please use cell id in first column and give ata least 2 columns.")
                return
            meta_cell_id = self.cell_meta.iloc[:,0].values
            meta_share, meta_barcode_index, share_barcode_index =\
                np.intersect1d(meta_cell_id,
                share_index, return_indices=True)
            _gene_barcode_index = gene_barcode_index[share_barcode_index]
            _atac_barcode_index = atac_barcode_index[share_barcode_index]
            if len(_gene_barcode_index) < 2: # no overlaps
                logger.info("Inconsistent metadata to expression data.")
                return
            tmp = joint_profiles["gene_barcodes"]
            joint_profiles["gene_barcodes"] = tmp.loc[_gene_barcode_index, :]
            temp = joint_profiles["atac_barcodes"]
            joint_profiles["atac_barcodes"] = temp.loc[_atac_barcode_index, :]

        else:
            # reorder rnaseq cell meta
            tmp = joint_profiles["gene_barcodes"]
            joint_profiles["gene_barcodes"] = tmp.loc[gene_barcode_index,:]
            temp = joint_profiles["atac_barcodes"]
            joint_profiles["atac_barcodes"] = temp.loc[atac_barcode_index, :]

        gene_tab = joint_profiles["gene_expression"]
        if issparse(gene_tab):
            joint_profiles["gene_expression"] = gene_tab[gene_barcode_index, :].A
        else:
            joint_profiles["gene_expression"] = gene_tab[gene_barcode_index, :]

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
            joint_profiles["atac_expression"] = _tmp.A
            joint_profiles["atac_names"] = joint_profiles["atac_names"].loc[high_count_atacs_index[1], :]

        else:
            _tmp = reorder_atac_exp[:, high_count_atacs]

            joint_profiles["atac_expression"] = _tmp
            joint_profiles["atac_names"] = joint_profiles["atac_names"].loc[high_count_atacs, :]

         # RNA-seq as the key
        Ys = []
        measurement = CellMeasurement(
            name="atac_expression",
            data=joint_profiles["atac_expression"],
            columns_attr_name="atac_names",
            columns=joint_profiles["atac_names"].astype(np.str),
        )
        Ys.append(measurement)
        # Add cell metadata
        if isinstance(self.cell_meta,pd.DataFrame):
            for l_index, label in enumerate(list(self.cell_meta.columns.values)):
                if l_index >0:
                    label_measurement = CellMeasurement(
                        name="{}_label".format(label),
                        data=self.cell_meta.iloc[meta_barcode_index,l_index],
                        columns_attr_name=label,
                        columns=self.cell_meta.iloc[meta_barcode_index, l_index]
                    )
                    Ys.append(label_measurement)
                    logger.info("Loading {} into dataset.".format(label))

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
                    logger.info("Unknown input data type:{}".format(_key))
                    return False
                # if not self.dataset[_key].split(".")[-1] in ["txt","tsv","csv"]:
                #     logger.debug("scMVP only support two files input of txt, tsv or csv!")
                #     return False
        elif len(self.dataset.keys()) >= 6:
            for _key in self._allow_input:
                if not _key in self.dataset.keys():
                    logger.info("Data type {} missing.".format(_key))
                    return False
        else:
            logger.info("Incorrect input file number.")
            return False
        for _key in self.dataset.keys():
            if not os.path.exists(self.data_path):
                logger.info("{} do not exist!".format(self.data_path))
            if not os.path.exists("{}{}".format(self.data_path, self.dataset[_key])):
                logger.info("Cannot find {}{}!".format(self.data_path, self.dataset[_key]))
                return False
        return True

    def _download(self, url: str, save_path: str, filename: str):
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

    def _add_cell_meta(self, cell_meta, filter=False):
        cell_ids = cell_meta.iloc[:,1].values
        share_index, meta_barcode_index, gene_barcode_index = \
            np.intersect1d(cell_ids,self.barcodes,return_indices=True)
        if len(share_index) <=1:
            logger.info("No consistent cell IDs!")
            return
        if len(share_index) < len(self.barcodes):
            logger.info("{} cells match metadata.".format(len(share_index)))
            return



class SnareDemo(LoadData):

    def __init__(self, dataset_name: str=None, data_path: str="/dataset"):
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126074"
        available_datasets = {
            "CellLineMixture": {
                "gene_expression": "GSE126074_CellLineMixture_SNAREseq_cDNA_counts.tsv.gz",
                "atac_expression": "GSE126074_CellLineMixture_SNAREseq_chromatin_counts.tsv.gz",
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
            super(SnareDemo, self).__init__(dataset = available_datasets[dataset_name],
                         data_path= data_path,
                         dense = False,
                         measurement_names_column = 1,
                         remove_extracted_data = False,
                         delayed_populating = False,
                         file_separator = "\t",
                         gzipped = True,
                         atac_threshold = 0.0005,
                         cell_threshold = 1
                         )
        elif dataset_name=="AdBrainCortex" or dataset_name=="P0_BrainCortex":
            super(SnareDemo, self).__init__(dataset=available_datasets[dataset_name],
                             data_path=data_path,
                             dense=False,
                             measurement_names_column=1,
                             remove_extracted_data=False,
                             delayed_populating=False,
                             gzipped=True,
                             atac_threshold=0.0005,
                             cell_threshold=1
                             )
        else:
            logger.info('Please select from "CellLineMixture", "AdBrainCortex" or "P0_BrainCortex" dataset.')


class PairedDemo(LoadData):

    def __init__(self, dataset_name: str = None, data_path: str = "/dataset"):
        urls = [
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130399/suppl/GSE130399_GSM3737488_GSM3737489_Cell_Mix.tar.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130399/suppl/GSE130399_GSM3737490-GSM3737495_Adult_Cerebrail_Cortex.tar.gz",
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130399/suppl/GSE130399_GSM3737496-GSM3737499_Fetal_Forebrain.tar.gz"
                ]

        available_datasets = {
            "CellLineMixture": {
                "gene_names": "Cell_Mix_RNA/genes.tsv",
                "gene_expression": "Cell_Mix_RNA/matrix.mtx",
                "gene_barcodes": "Cell_Mix_RNA/barcodes.tsv",
                "atac_names": "Cell_Mix_DNA/genes.tsv",
                "atac_expression": "Cell_Mix_DNA/matrix.mtx",
                "atac_barcodes":"Cell_Mix_DNA/barcodes.tsv"
            },
            "Adult_Cerebral": {
                "gene_names": "Adult_CTX_RNA/genes.tsv",
                "gene_expression": "Adult_CTX_RNA/matrix.mtx",
                "gene_barcodes": "Adult_CTX_RNA/barcodes.tsv",
                "atac_names": "Adult_CTX_DNA/genes.tsv",
                "atac_expression": "Adult_CTX_DNA/matrix.mtx",
                "atac_barcodes": "Adult_CTX_DNA/barcodes.tsv"
            },
            "Fetal_Forebrain": {
                "gene_names": "FB_RNA/genes.tsv",
                "gene_expression": "FB_RNA/matrix.mtx",
                "gene_barcodes": "FB_RNA/barcodes.tsv",
                "atac_names": "FB_DNA/genes.tsv",
                "atac_expression": "FB_DNA/matrix.mtx",
                "atac_barcodes": "FB_DNA/barcodes.tsv"
            }
        }

        if dataset_name=="CellLineMixture" or dataset_name=="Fetal_Forebrain":
            if os.path.exists("{}/Cell_embeddings.xls".format(data_path)):
                cell_embed = pd.read_csv("{}/Cell_embeddings.xls".format(data_path), sep='\t')
                cell_embed_info = cell_embed.iloc[:, 0:2]
                cell_embed_info.columns = ["Cell_ID","Cluster"]
            else:
                logger.info("Cannot find cell embedding files for Paried-seq Demo.")
                return
            super().__init__(dataset = available_datasets[dataset_name],
                             data_path= data_path,
                             dense = False,
                             measurement_names_column = 1,
                             remove_extracted_data = False,
                             delayed_populating = False,
                             gzipped = False,
                             atac_threshold = 0.005,
                             cell_threshold = 100,
                             cell_meta=cell_embed_info
                             )
        elif dataset_name=="Adult_Cerebral":
            if os.path.exists("{}/Cell_embeddings.xls".format(data_path)):
                cell_embed = pd.read_csv("{}/Cell_embeddings.xls".format(data_path), sep='\t')
                cell_embed_info = cell_embed.iloc[:, ["ID","Cluster"]]
                cell_embed_info.columns = ["Cell_ID","Cluster"]
            else:
                logger.info("Cannot find cell embedding files for Paried-seq Demo.")
                return
            super().__init__(dataset=available_datasets[dataset_name],
                             data_path=data_path,
                             dense=False,
                             measurement_names_column = 1,
                             remove_extracted_data=False,
                             delayed_populating=False,
                             gzipped=False,
                             atac_threshold=0.005,
                             cell_threshold=100,
                             cell_meta=cell_embed_info
                             )
        else:
            logger.info('Please select from {} dataset.'.format("\t".join(available_datasets.keys())))


class SciCarDemo(LoadData):
    def __init__(self, dataset_name: str = None, data_path: str = "/dataset"):
        urls = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE117089&format=file"
        # NOTICE, tsv files are generated from original txt files
        available_datasets = {
            "CellLineMixture": {
                "gene_barcodes": "GSM3271040_RNA_sciCAR_A549_cell.tsv",
                "gene_names": "GSM3271040_RNA_sciCAR_A549_gene.tsv",
                "gene_expression": "GSM3271040_RNA_sciCAR_A549_gene_count.txt",
                "atac_barcodes": "GSM3271041_ATAC_sciCAR_A549_cell.tsv",
                "atac_names": "GSM3271041_ATAC_sciCAR_A549_peak.tsv",
                "atac_expression": "GSM3271041_ATAC_sciCAR_A549_peak_count.txt"
            },
            "mouse_kidney": {
                "gene_barcodes": "GSM3271044_RNA_mouse_kidney_cell.tsv",
                "gene_names": "GSM3271044_RNA_mouse_kidney_gene.tsv",
                "gene_expression": "GSM3271044_RNA_mouse_kidney_gene_count.txt",
                "atac_barcodes": "GSM3271045_ATAC_mouse_kidney_cell.tsv",
                "atac_names": "GSM3271045_ATAC_mouse_kidney_peak.tsv",
                "atac_expression": "GSM3271045_ATAC_mouse_kidney_peak_count.txt"
            }
        }
        if dataset_name:
            for barcode_file in ["gene_barcodes", "atac_barcodes", "gene_names", "atac_names"]:
                # generate gene and atac barcodes from cell metadata.
                with open("{}/{}".format(data_path, available_datasets[dataset_name][barcode_file]),"w") as fo:
                    infile = "{}/{}.txt".format(data_path, available_datasets[dataset_name][barcode_file][:-4])
                    indata = [i.rstrip().split(",") for i in open(infile)][1:]
                    for line in indata:
                        fo.write("{}\n".format(line[0]))

            super().__init__(dataset=available_datasets[dataset_name],
                             data_path=data_path,
                             dense=False,
                             measurement_names_column=0,
                             remove_extracted_data=False,
                             delayed_populating=False,
                             gzipped=False,
                             atac_threshold=0.0005,
                             cell_threshold=1
                             )
        else:
            logger.info('Please select from {} dataset.'.format("\t".join(available_datasets.keys())))

