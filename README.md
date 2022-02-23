# Datasets Briefs for AntennaVAE

## 1. FOLDER NAME: Pancreas

The Pancreas dataset is a collection of several publicly available human pancreas datasets. <font color = red>Don't have time information</font>
|Batch ID|Cell Number|
|-|-|
|0|1004|
|1|2285|
|2|638|
|3|2394|
|4|1937|
|5|1724|
|6|3605|
|7|1303|

## 2. FOLDER NAME: scp_gex_matrix

Except for Control, there are 6 different cohorts in this dataset, corresponds to 
different severity of sepsis, from “Leuk-UTI” to “ICU-NoSEP” is mildest to most 
severe, and here different cohorts represent different time point.  


|Label|0|1|2|3|4|5|6|
|-|-|-|-|-|-|-|-|
|Cohort|Control|Leuk-UTI|Int-URO|URO|Bac-SEP|ICU-SEP|ICU-NoSEP|
|Cells|48855|22234|13875|16136|5441|7997|8019|
|Batch ID(1-35)|1,2|3, 5|7, 11|13, 15|31, 33|27, 29|19, 21|
|Cell in Batch|4566, 705|1783, 4646|1981, 1848|3465, 1573|3229, 834|500, 4754|2323, 1406|

<font color = red>Have 35 batches in total, but may overlap across cohorts, for details please check the csv file</font>
```
/project/shared/AntennaVAE_datasets/scp_gex_matrix/processed_sepsis_7533/sepsis_processed_batch_info.csv
```

For testing the model, I fisrt filtered cohort names, and for each name filtered batch id 1-35, stored in a separate files, and selected 14 from them, 2 for each cohort, making sure no overlapping across batches(i.e. if batch 1 contain cohort A and B, I only use batch 1 contain A, and choose another one for cohortB)

<font color = orange>Our primary cohorts targeted patients with urinary tract infection (UTI) early in their disease course, within 12 hours of presentation to the Emergency Department (ED)</font>

**According to the paper, the blue ones are primiary cohorts, which targeted patients with urinary tract infection (UTI) early in their disease course, within 12 hours of presentation to the Emergency Department (ED)**

**The green ones are secondary cohorts, which are later in their disease course, enrolled at least 24 hours after initial hospital presentation and receipt of intravenous antibiotics**

**The red one is controls**

|Name|Interpration|
|-|-|
|<font color = red>Control</font>| uninfected, healthy controls|
|<font color = cyan>Leuk-UTI</font>|UTI with leukocytosis (blood WBC ≥ 12,000 per mm3) but no organ dysfunction (Leuk-UTI)|
|<font color = cyan>Int-URO</font>|UTI with mild or transient organ dysfunction|
|<font color = cyan>URO</font>|UTI with organ dysfunction|
|<font color = lime>Bac-SEP</font>|bacteremic patients with sepsis in hospital wards|
|<font color = lime>ICU-SEP</font>|patients admitted to the ICU with sepsis|
|<font color = lime>ICU-NoSEP</font>|patients admitted to the ICU without sepsis|


## 3. Dataset from Single-cell mapping paper, FOLDER NAME: single_cell_mapping

There are two groups of data:  
1. iep group:contains two time courses of data with 5 time points in it.  
2. reprogramming group: have 8 batches and 2 "HF", <font color = yellow>I'm not quite sure about what's the meaning of HF</font>

The details of the files can be found in 
```
/project/shared/AntennaVAE_datasets/single_cell_mapping/batch_info.csv
```

<!-- They use [singleCellNet](https://github.com/pcahan1/CellNet) to cluster the data and I think they don't have batch information

|Day 4|Day 6|
|-|-|
|(5062, 15564)|(4842, 15564)|
|8 clusters|10 clusters| -->