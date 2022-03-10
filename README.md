# Code for experiments on measuring domain similarity based on embedding spaces

## Embedding spaces
The embedding spaces used in this study were trained using the [PPMI+SVD implementation from Levy et al. (2015)](
https://bitbucket.org/omerlevy/hyperwords/src/f5a01ea3e44c/). The English 260 MB embeddings are contained in the embeddings directory. The complete set of pre-trained embedding spaces can be downloaded via
```
wget https://cis.lmu.de/beyera/embeddings/embeddings.zip
```

Other resources (e.g. vocab files, corpora and embeddings from the simulation study) are available upon request. (anne.beyer@campus.lmu.de)

## Requirements
CCA measure: </br>
  - python3 </br>
  - numpy </br>
  - sklearn </br>
  - gensim </br>
 
Simulation study: </br>
The PPMI+SVD embeddings in the simulation study require Python 2.7 (see link above for further dependencies)</br>
As the other project scripts are written in Python 3.6, conda is used to switch envronments (ebeddings and embeddings2) in ```corpus_simulation.sh```

## CCA Measure
```mapping_correlation.sh``` assumes a directory "embeddings" containing the pre-trained embedding spaces (adapt paths as appropriate) and creates a directory "correlations", in which it computes the correlation scores for all corpus combinations described in the paper, as well as creating a visualization of the dimension-wise correlations. The CCA measure scores are printed to stdout.

## Simulation study
TODO
