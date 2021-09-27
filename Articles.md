# scClass: A immune cell classifier tool create by supervised deep learning

## Abstract
Single-cell RNA sequencing(scRNA-seq) is a novel RNA seqensing method which can track their RNA-expression in every single cell. 
However, the traditional way to annotate cell-type like isolate by flow cytometry or clustering by seurat is either expensive or inefficient.
Here we present scClass, a supervised deep learning model for annotating celltype on immmune cell.
scClass use bone marrow and cord blood dataset to training and try to predict on pbmc and other dataset.
We provide a package for running scClass on Python and have a demo on Colab at .....

## Introduction
Cell type annotation is one of the most important step in Single-cell RNA sequencing analysing.
There are many annotation tools which can found on scRNA-tools nowadays.
As the amount of cells in one scRNA-seq study grow up immediately,
but most scRNA-tools aren't optimized, so that it cost more memory and computation.
In our demo, we classify around 140,000 cells but use only 12GB moemory on Colab.

## Dataset and Model
scClass is a Python package that allows 
we use 10X genomics datasets since it is a popularization scRNA-seq method.
