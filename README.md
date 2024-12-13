# The Replicate of GenoMap and TCER

The success of transcriptomic studies depends critically on the accuracy of the gene expression counts. In practice, gene expression data often suffer from low transcript capture efficiency and technical noise, leading to inaccurate gene counts. Thus, recovery of the expression values of the genes using computational techniques is critical for the downstream applications. Existing methods of gene expression recovery techniques suffer from either accuracy or computational efficiency or both ,which limits their practical applications. In reality, it has long been recognized that gene–gene interactions may serve as reflective indicators of underlying biology processes, presenting discriminative signatures of the cells. A genomic data analysis framework that is capable of leveraging the underlying gene–gene interactions is thus highly desirable and could allow for more reliable identification of distinctive patterns of the genomic data through extraction and integration of intricate biological characteristics of the genomic data. Based on the above questions, Wei et al. leverage the interactive information and establish a transform-and-conquer expression recovery (TCER) strategy to tackle the gene imputation problem. 

## TCER
TCER is a self-supervised deep learning framework for gene expression recovery. The proposed pipeline consists of two steps, inluding (i) we first reposition the genes in such a way that their spatial configuration reflects their interactive relationships; and (ii) we then use a self-supervised 2D convolutional neural network (ER-Net) to extract the contextual features of the interactions from the spatially configured genes and impute the omitted values. 

## Structure of TCER

### Pipeline
<img src="imgs/pipeline.png" style="zoom:38%;" />

Pipeline of the proposed TCER as shown in the first picture. 1D gene expression data is first converted into an image format where the gene–gene interactions are reflected naturally in the spatial configuration of GenoMap. A dropout simulation strategy is then applied to simulate the dropout events where non-zero values are randomly masked. Last, the proposed ER-Net is employed to impute the masked GenoMap. The GenoMap construction is critical in TCER for the network to recover the gene expression values.


### ER-Net Structure
<img src="imgs/network_structure.png" style="zoom:38%;" />

To extract deep interaction information from a GenoMap for the recovery of missing expression values, a novel encoderdecoder architecture referred to as expression recovery network (ER-Net) is designed. Network structure of ER-Net as shown in the second picture. In figure(A) ER-Net follows an encoder-decoder structure and employs three cascaded DFA module with deformable convolution to extract both local and global features of gene–gene interactions. In figure(B) Detailed structure of the proposed DFA module.

## GenoMap

Genomap is an entropy-based cartography strategy to contrive the high dimensional gene expression data into a configured image format with explicit integration of the genomic interactions. This unique cartography casts the gene-gene interactions into a spatial configuration and enables us to extract the deep genomic interaction features and discover underlying discriminative patterns of the data. For a wide variety of applications (cell clustering and recognition, gene signature extraction, single-cell data integration, cellular trajectory analysis, dimensionality reduction, and visualization), genomap drastically improves the accuracy of data analyses as compared to state-of-the-art techniques.

## Structure of GenoMap
<img src="imgs/GenoMap.png" style="zoom:38%;" />

To maximally reflect the gene-gene interaction information of the system through a 2D spatial configuration of genes, it transform the dataset into a series of cell-specific genomaps by optimizing a transport function. As the possible ways of gene placement into a 2D grid for a cell is a factorial of the number of involved genes, a robust optimization of the transport function is imperative to reliably construct a genomap. In general, a genomap possesses the basic characteristics of an image with the pixelated configuration manifesting the gene-gene interactions and provides a comprehensive representation of the gene expression data. After the construction of the genomaps, we extract the configurational features of the genomic interactions by using an efficient convolutional neural network (CNN) named genoNet. In this way, deep correlative features of the genes are extracted effectively from the data for subsequent decision-making.  For a wide variety of applications, including cell clustering and recognition, gene signature extraction, single cell data integration, cellular trajectory analysis, dimensionality reduction, and visualization, the proposed approach substantially outperforms the state-ofthe-art methods.

## Replicate work

### Data 

To run the model, you will need to download data files```TM_data.csv``` from [here](https://drive.google.com/drive/folders/1xq3bBgVP0NCMD7bGTXit0qRkL8fbutZ6?usp=drive_link), download the ```TMdata.mat``` from [here](https://github.com/xinglab-ai/genomap/blob/main/MATLAB/TMdata.mat)


### 1. Transform the data by GenoMap













