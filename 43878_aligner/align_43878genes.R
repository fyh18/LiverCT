library(Seurat)
library(data.table)
library(ggplot2)
suppressMessages(suppressWarnings(library(stringr)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(library(data.table))

#### Read Data ####
dataPath <- "/home/wyh/liver_atlas/submit/revision/"
author <- "Han"

refTablePath <- paste0("/home/wyh/liver_atlas/code/LiverCT/43878_aligner/GeneSymbolRef_SelectAll_upd.csv")
refGenePath <- paste0("/home/wyh/liver_atlas/code/LiverCT/43878_aligner/total_gene_list_43878.txt")

# expression matrix
data.matrix <- read.table("/home/wyh/liver_atlas/submit/revision/Han/Adult-Liver4_dge.txt", sep=',',row.names = 1,header = TRUE)
dataobj <- CreateSeuratObject(counts = data.matrix, 
                              project = author)
rm(data.matrix)
gc()

#### Transfer genes ####
ref_table <- read.csv(refTablePath, header=TRUE, na.strings=TRUE, stringsAsFactors=FALSE)
total_gene_list <- read.table(refGenePath, header=TRUE, sep='\t', fill=TRUE, stringsAsFactors=FALSE)
total_gene_list = total_gene_list[,1]
total_gene_list <- str_replace(total_gene_list, "_", "-")

ref_table <- ref_table[,c("Approved.symbol","Alias.symbol","Previous.symbol")]
ref_table <- ref_table[ref_table[,"Previous.symbol"]!="" | ref_table[,"Alias.symbol"]!="",]
# Seurat changes all "_" to "-".
ref_table$Previous.symbol <- str_replace(ref_table$Previous.symbol, "_", "-")
ref_table$Alias.symbol <- str_replace(ref_table$Alias.symbol, "_", "-")
ref_table$Approved.symbol <- str_replace(ref_table$Approved.symbol, "_", "-")
ref_table_prev <- unique(ref_table[,c("Approved.symbol","Previous.symbol")])
ref_table_prev <- ref_table_prev[ref_table_prev[,"Previous.symbol"]!="",]
ref_table_alia <- unique(ref_table[,c("Approved.symbol","Alias.symbol")])
ref_table_alia <- ref_table_alia[ref_table_alia[,"Alias.symbol"]!="",]

# *--------------------Load query data--------------------*
print("=========Loading Query Data=========")
result_data <- as.data.frame(as.matrix(dataobj@assays$RNA@data))
query_gene_list <- rownames(dataobj)

rm(ref_table, dataobj)
gc()

# *--------------------Perform gene name uniform--------------------*
print("=========Performing Gene Symbol Uniform=========")
print("Performing gene symbol uniform, this step may take several minutes")
gene_appearance_list <- data.frame(gene_name=total_gene_list, appearance=rep(FALSE, length(total_gene_list)))
outlier_gene_list <- c()

result_data$genenames <- rownames(result_data)
report <- data.frame(Original.Name=character(), Modified.Name=character(), Status=character(), stringsAsFactors=FALSE)

for (i in c(1:length(query_gene_list))){
    gene_name <- query_gene_list[i]
    # Modify gene symbols, both "Alias symbol" and "Previous symbol" are used. 
    if(sum(ref_table_prev["Previous.symbol"]==gene_name)>0){
        # Multiple names matched
        if(sum(ref_table_prev["Previous.symbol"]==gene_name)>1){
            candidate_names <- paste(ref_table_prev[ref_table_prev[,"Previous.symbol"]==gene_name,"Approved.symbol"], collapse='|')
            report[i,] <- c(gene_name, candidate_names, "Multiple Candidates")
            if(gene_name %in% total_gene_list){
                gene_appearance_list[gene_appearance_list[,"gene_name"]==gene_name,"appearance"] = TRUE
            }
            else{
                outlier_gene_list <- c(outlier_gene_list, gene_name)
            }
        }
        # Only one name matched
        else{
            candidate_names <- ref_table_prev[ref_table_prev[,"Previous.symbol"]==gene_name,"Approved.symbol"]
            result_data[i:dim(result_data)[1],"genenames"][result_data[i:dim(result_data)[1],"genenames"]==gene_name] <- candidate_names
            report[i,] <- c(gene_name, candidate_names, "Changed")
            if(candidate_names %in% total_gene_list){
                gene_appearance_list[gene_appearance_list[,"gene_name"]==candidate_names,"appearance"] = TRUE
            }
            else{
                outlier_gene_list <- c(outlier_gene_list, candidate_names)
            }
        }
    }
    
    else if(sum(ref_table_alia["Alias.symbol"]==gene_name)>0){
        # Multiple names matched
        if(sum(ref_table_alia["Alias.symbol"]==gene_name)>1){
            candidate_names <- paste(ref_table_alia[ref_table_alia[,"Alias.symbol"]==gene_name,"Approved.symbol"], collapse='|')
            report[i,] <- c(gene_name, candidate_names, "Multiple Candidates")
            if(gene_name %in% total_gene_list){
                gene_appearance_list[gene_appearance_list[,"gene_name"]==gene_name,"appearance"] = TRUE
            }
            else{
                outlier_gene_list <- c(outlier_gene_list, gene_name)
            }
        }
        # Only one name matched
        else{
            candidate_names <- ref_table_alia[ref_table_alia[,"Alias.symbol"]==gene_name,"Approved.symbol"]
            result_data[i:dim(result_data)[1],"genenames"][result_data[i:dim(result_data)[1],"genenames"]==gene_name] <- candidate_names
            report[i,] <- c(gene_name, candidate_names, "Changed")
            if(candidate_names %in% total_gene_list){
                gene_appearance_list[gene_appearance_list[,"gene_name"]==candidate_names,"appearance"] = TRUE
            }
            else{
                outlier_gene_list <- c(outlier_gene_list, candidate_names)
            }
        }
    }
    
    # Gene name not found
    else{
        report[i,] <- c(gene_name, gene_name, "No Change")
        if(gene_name %in% total_gene_list){
            gene_appearance_list[gene_appearance_list[,"gene_name"]==gene_name,"appearance"] = TRUE
        }
        else{
            outlier_gene_list <- c(outlier_gene_list, gene_name)
        }
    }
}

# *--------------------Construct uniform output--------------------*
print("=========Building Output Matrix=========")
setDT(result_data)
result_data_sub <- result_data[,lapply(.SD, mean, na.rm=TRUE),by=genenames,
                                   .SDcols=names(result_data)[1:(dim(result_data)[2]-1)]]
result_data_sub <- as.data.frame(result_data_sub)[which(!result_data_sub$genenames %in% outlier_gene_list),]
result_data_out <- subset(result_data_sub, select = -genenames )
rownames(result_data_out) <- result_data_sub$genenames
print("Shape of processed query data: ")
print(dim(result_data_out))

add_df <- data.frame(matrix(nrow=sum(gene_appearance_list$appearance==FALSE),ncol=dim(result_data)[2]-1, 0))

rm(result_data, result_data_sub)
gc()

rownames(add_df) <- gene_appearance_list$gene_name[!gene_appearance_list$appearance]
colnames(add_df) <- colnames(result_data_out)

result_data_out <- rbind(result_data_out, add_df)
# order rows from gene symbols A to Z
# result_data_out<-result_data_out[order(row.names(result_data_out)),]
result_data_out<-result_data_out[total_gene_list,]

#### Save the expression matrix ####
write.table(result_data_out, 
            file.path(dataPath, author, 
                      paste0(author, "_expression_43878.tsv")), 
            quote = F, sep = "\t", 
            col.names = T, row.names = T)
