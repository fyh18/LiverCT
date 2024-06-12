# A tutorial for 43878_aligner
### gene alignment

> Note: Data should be processed in RDS format

```
cd LiverCT/43878_aligner/ 进入子文件夹
Rscript align_43878genesV2.R data_dir data_name
Eg. Rscript align_43878genesV2.R /home/data/Lu2022/ Lu2022
```


### convert to h5ad file

```
Expr_matrix = pd.read_table("/home/data/Lu2022/Lu2022_expression_43878.tsv")
ad1 = sc.AnnData(X= Expr_matrix.T)
```
