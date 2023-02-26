# Struct2GO:protein function prediction based on Graph pooling algorithm and AlphaFold2 structure information
# Abstract
Struct2GO is a protein function prediction model based on self-attention graph pooling, which utilizes structural information from AlphaFold2 to augment the accuracy and generality of the model's predictions.


![avatar](/model.png)

# Data
- Protein structure: download from the [AlphaFold Protein Struct Databasee](https://alphafold.ebi.ac.uk/download)
- Protein sequence: download from the [UniProt website](https://www.uniprot.org/) 
- Protein annotion: down from the [GOA website](https://www.ebi.ac.uk/GOA/)
- Gene Ontology: download from the [GO website](http://geneontology.org/)
  
We put the processed data for train and test on [there](https://github.com/lyjps/Struct2GO/tree/master/divided_data)

# Usage
## Train the model
Run the ``run_train.sh`` script directly to train the model(e.g. for MFO)
 ```python
 python run_train.sh
 ``` 

Note: Remember to update the file directory in the script to your local directory if you wish to run the MFO model or the other two models.

## Evaluation the model
Run the ``run_test.sh`` scirpy directly to evaluation the model(e.g. for MFO)
``` python
python run_test.sh
```

Note: Remember to update the file directory in the script to your local directory if you wish to evaluation the MFO model or the other two models.

## Processing raw data
we provide the proccesed data for training and evaluating directly [there](https://github.com/lyjps/Struct2GO/tree/master/divided_data), and then we will explain how to process the raw data.
### Protein struction data
- Download protein structure data and convert the three-dimensional atomic structure of proteins into protein contact maps.
```
cd ./data_processing
python predicted_protein_struct2map.py
```
- Obtain amino acid residue-level features through the Node2vec algorithm.
```
cd ./angel-master/spark-on-angel/example/local/Node2VecExample.scala
```
(ps:run it by the IntelLLiJ IDEA )
```
cd .data_processing
python sort.py
```

### Protein sequence data
- Download protein sequence data obtain protein sequence features through the Seqvec model.
```
cd ./data_processing
python seq2vec.py
```

### Fuse protein structure and sequence data and divide the dataset
```
cd ./model
python labels_load.p
cd ./data_processing
python divide_data.py
```






