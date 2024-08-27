# !/usr/bin/python
# -*- coding: utf-8 -*-
# Pytoolkit GeneSymbolUniform for hECA 2.0
# author: Yixin Chen, Haiyang Bian
# date: 2023/09/25

import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse 
import pathlib

# functions used to construct the sparse matrix parallelly
def build_partial_csr(data_values, data_rows, data_cols, shape):
    return sp.csr_matrix((data_values, (data_rows, data_cols)), shape=shape)

def parallel_sparse_matrix(result_data_values, result_data_rows, result_data_cols, n_row, n_col, num_threads=4):
    chunk_size = len(result_data_values) // num_threads

    matrices = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in tqdm(range(num_threads), desc="Building matrices"):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != num_threads - 1 else len(result_data_values)

            futures.append(executor.submit(
                build_partial_csr, 
                result_data_values[start:end],
                result_data_rows[start:end],
                result_data_cols[start:end],
                (n_row, n_col)
            ))

        for future in tqdm(futures, desc="Collecting results"):
            matrices.append(future.result())

    return sum(matrices)

# the function to uniform gene expression matrix and save it into Anndata
def GeneSymbolUniform(input_adata=None, input_path=None, ref_table_path="./GeneSymbolRef_SelectAll_upd.csv", 
                      gene_list_path="./total_gene_list_43878.txt", 
                      output_dir="./", output_prefix = "",
                      print_report=True, average_alias = True,
                      n_threads=10):
    """
    This function can be divided into following steps: 
        1) load reference table of the symbol relationships; 
        2) load the quert data (from a scanpy h5ad file with counts in 'X'); 
        3) construction the mapping dict between approved symbols and query symbols; 
        4) construct the output h5ad file and save the file and report
        
    
    :param input_adata: the input h5ad data with X as the count matrix to be uniformed.
    :type input_adata: scanpy::AnnData
    :default input_path: None
    
    :param input_path: the path of the input h5ad file (only used when input_data was not given)
    :type input_path: str
    :default input_path: None
    
    :param ref_table_path: the path of the reference table
    :type ref_table_path: str
    :default ref_table_path: "./GeneSymbolRef_SelectAll_upd.csv"
    
    :param gene_list_path: the path of the total gene list table
    :type gene_list_path: str
    :default gene_list_path: "./total_gene_list_43878.txt"
    
    :param output_dir: the path to save the output h5ad file
    :type output_dir: str
    :default output_dir: "./"
    
    :param output_prefix: the prefix of the output and report files
    :type output_prefix: str
    :default output_prefix: ""
    
    :param print_report: if print a report of the modified genes in the report.csv under the output_dir
    :type print_report: bool
    :defaul print_report: True
    
    :param average_alias: if average the counts of the genes mapped to the same aprroved symbol
    :type average_alias: bool
    :default average_alias: True
    
    
    :return: a h5ad data with uniformed epxression matrix.
    :rtype: scanpy::AnnData
    """
    # print("=========Task Started!=========")
    # *--------------------Load reference table--------------------*
    # print("=========Loading Reference Table=========")
    # Read the CSV
    ref_table_raw = pd.read_csv(ref_table_path, dtype=str,index_col=0)

    # print("Finished")
    # *--------------------Data processing--------------------*
    # print("=========Processing Reference Table=========")

    # Select necessary columns
    ref_table_raw = ref_table_raw.loc[ref_table_raw.Status == 'Approved',:]
    ref_table_raw = ref_table_raw[["Approved symbol", "Alias symbol", "Previous symbol"]]

    # Filter rows with non-empty "Previous symbol" and "Alias symbol"
    ref_table_raw = ref_table_raw[(ref_table_raw["Previous symbol"].notna() & ref_table_raw["Previous symbol"] != "") |
                                  (ref_table_raw["Alias symbol"].notna() & ref_table_raw["Alias symbol"] != "")]

    # Replace "_" with "-"
    ref_table_raw["Previous symbol"] = ref_table_raw["Previous symbol"].str.replace("_", "-")
    ref_table_raw["Alias symbol"] = ref_table_raw["Alias symbol"].str.replace("_", "-")
    ref_table_raw["Approved symbol"] = ref_table_raw["Approved symbol"].str.replace("_", "-")

    # Create subsets for Previous and Alias symbols
    ref_table_prev = ref_table_raw[["Approved symbol", "Previous symbol"]].drop_duplicates()
    ref_table_prev = ref_table_prev[ref_table_prev["Previous symbol"].notna() & ref_table_prev["Previous symbol"] != ""]

    ref_table_alia = ref_table_raw[["Approved symbol", "Alias symbol"]].drop_duplicates()
    ref_table_alia = ref_table_alia[ref_table_alia["Alias symbol"].notna() & ref_table_alia["Alias symbol"] != ""]

    # print("Finished")

    # *--------------------Load query data--------------------*
    # print("=========Loading Query Data=========")
    if input_path is None and input_adata is None:
        # Neither of the input data form is given
        print("Error: No input data is given.")
        return(-1)
    elif not input_adata is None:
        # if adata is given, the input_path will be igored.
        adata_input = input_adata
    else:
        adata_input = sc.read_h5ad(input_path)
        
    query_data = adata_input.X
    query_gene_list = np.array(adata_input.var_names)
    # print("The shape of query data is: {} ".format(query_data.shape))
    # print("Print out first 5 genes in query data, in case something wrong happens in data loading: ")
    # print(query_gene_list[:5])
    # print("Finished")

    # *--------------------Load total gene list--------------------*
    # print("=========Processing Gene List=========")

    # Read the gene list from a tab-separated file
    total_gene_list_raw = pd.read_csv(gene_list_path,
                                      sep='\t', header=0, na_values=pd.NA)

    # Select the first column as the gene list
    total_gene_list = total_gene_list_raw.iloc[:, 0]

    # Replace all "_" with "-"
    total_gene_list = total_gene_list.str.replace("_", "-")
    total_gene_list = total_gene_list.values
    total_gene_list.sort()

    # Print the length of the gene list
    # print("The length of reference gene_list is: {}".format(len(total_gene_list)))
    # print("Finished")


    # *--------------------Performing Gene Symbol Uniform--------------------*
    # print("=========Performing Gene Symbol Uniform=========")
    print("Performing gene symbol uniform, this step may take several minutes")

    # Create a DataFrame to store gene appearances
    gene_appearance_dict = {key: [] for key in total_gene_list}
    outlier_gene_list = []

    # build a DataFrame for report
    report = pd.DataFrame(columns=['Original.Name', 'Modified.Name', 'Status'])
    report['Original.Name'] = query_gene_list

    for query_symbol in tqdm(query_gene_list, desc="Processing"):
        if query_symbol in total_gene_list:
            # if the query gene is in the approved list, we donot change it
            gene_appearance_dict[query_symbol].append(query_symbol)
            report.loc[report['Original.Name'] == query_symbol, 'Status'] = "No Change"
        else:
            # Matching query symbols with Alias symbol
            candidates_A = ref_table_alia.loc[ref_table_alia['Alias symbol'] == query_symbol,"Approved symbol"].values
            # Matching query symbols with Previous symbol
            candidates_P = ref_table_prev.loc[ref_table_prev['Previous symbol'] == query_symbol,"Approved symbol"].values

            candidates = np.unique(np.hstack([candidates_A,candidates_P]))

            if len(candidates)>1:
                # one input symbols with multiple aprroved symbols
                report.loc[report['Original.Name'] == query_symbol, 'Modified.Name'] = '|'.join(candidates)
                report.loc[report['Original.Name'] == query_symbol, "Status"] = "Multiple Candidates"
                outlier_gene_list.append(query_symbol)
            elif len(candidates)==1:
                gene_appearance_dict[candidates[0]].append(query_symbol)
                report.loc[report['Original.Name'] == query_symbol, 'Modified.Name'] = candidates
                report.loc[report['Original.Name'] == query_symbol, "Status"] = "Changed"
            else:
                # not found in the Alias symbol and Previous symbol
                outlier_gene_list.append(query_symbol)
                report.loc[report['Original.Name'] == query_symbol, "Status"] = "Abandoned"

    gene_appearance_dict_filtered =  {key: value for key, value in gene_appearance_dict.items() if len(value) > 0}
    print("Finished")


    # *--------------------Build output data--------------------*
    # print("=========Building output data=========")
    # print("Building output data, this step may take several minutes")

    n_row = query_data.shape[0]
    n_col = len(total_gene_list)
    result_data_values = []
    result_data_rows = []
    result_data_cols = []

    total_gene_list_indices = {gene: index for index, gene in enumerate(total_gene_list)}
    query_gene_list_indices = {gene: index for index, gene in enumerate(query_gene_list)}

    query_data_csc = query_data.tocsc()  # Convert to CSC format for efficient column operations

    for key, source_genes in tqdm(gene_appearance_dict_filtered.items(), desc="Processing"):
        target_col = total_gene_list_indices.get(key)
        source_col = np.unique([query_gene_list_indices[gene] for gene in source_genes])

        col_sum = query_data_csc[:, source_col].sum(axis=1).A.flatten()  # Sum the values
        
        if average_alias:
            # this step only used when average_alias is set as True
            if len(source_col) > 1:
                col_sum = np.round(col_sum / len(source_col))

        # Add data to temporary storage
        non_zero_rows = np.where(col_sum != 0)[0]
        result_data_values.extend(col_sum[non_zero_rows])
        result_data_rows.extend(non_zero_rows)
        result_data_cols.extend([target_col] * len(non_zero_rows))

    # Construct the sparse matrix from the three lists
   
    if n_row >= 1e5:
        print("Because the number of cells is very large, we construct the sparse matrix parallelly with {} threads".format(n_threads))
        result_data = parallel_sparse_matrix(result_data_values, result_data_rows, result_data_cols, n_row, n_col,n_threads)
    else:
        result_data = sp.csr_matrix((result_data_values, (result_data_rows, result_data_cols)), shape=(n_row, n_col))
    # print("Shape of output data is {}. It should have 42117 genes with cell number unchanged.".format(result_data.shape))
    
     # *--------------------Build output data--------------------*
    # print("=========Building output h5ad=========")
    adata_output = sc.AnnData(X=result_data,obs=adata_input.obs,var=pd.DataFrame(index=total_gene_list))
    # print("Finished")
    # print("=========Saving output h5ad=========")
    # adata_output.write(os.path.join(output_dir,"{}_uniformed.h5ad".format(output_prefix)))
    # print("h5ad file saved in:", os.path.join(output_dir,"{}_uniformed.h5ad".format(output_prefix)))
    # print("Finished")
    
    if print_report:
        print("=========Printing report=========")
        report.to_csv(os.path.join(output_dir,"{}_report.csv".format(output_prefix)))
        print("report file saved in:", os.path.join(output_dir,"{}_report.csv".format(output_prefix)))
        print("Finished")
        
    return(adata_output)


    
if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description='GeneSymbolUniform parameters')
    parser.add_argument('--input_path', type=str, help='the path of the input h5ad file')
    parser.add_argument('--ref_table_path', default="./GeneSymbolRef_SelectAll_upd0731.csv", type=str, help='the path of the reference table.')
    parser.add_argument('--gene_list_path', default="./total_gene_list_42117.txt", type=str, help='the path of the total gene list table.')
    parser.add_argument('--output_dir', default='./', type=str, help='the path to save the output h5ad file')
    parser.add_argument('--output_prefix', default='', type=str, help='the prefix of the output file and report file')
    parser.add_argument('--print_report', action='store_true', help='print a report of the modified genes in the report.csv under the output_dir')
    parser.add_argument('--average_alias', action='store_true', help='if average the counts of the genes mapped to the same aprroved symbol')
    parser.add_argument('--n_threads',  default=1,  type=int, help='number of threads used in the construction of sparse matrix')
    args = parser.parse_args()
    GeneSymbolUniform(input_path = args.input_path, 
                      ref_table_path = args.ref_table_path, 
                      gene_list_path = args.gene_list_path, 
                      output_dir = args.output_dir, 
                      output_prefix = args.output_prefix,
                      print_report = args.print_report, 
                      average_alias = args.average_alias,
                      n_threads = args.n_threads)