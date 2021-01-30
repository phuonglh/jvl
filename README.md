# Introduction

This software package implements some fundamental NLP tasks in the Julia programming language. Tasks which have been implemented so far include:

- word segmentation (for Vietnamese)
- part-of-speech tagging
- named entity recognition
- transition-based dependency parsing
- dependency graph embedding

All machine learning models are based on neural networks, either multi-layer perceptron (MLP) or recurrent neural networks (RNN) with LSTM and GRU variants, or attentional sequence to sequence models. 

# Part-of-Speech Tagging

An implementation of GRU models for part-of-speech tagging is provided in `seq/PoSTagger.jl`. The general pipeline is `EmbeddingWSP -> GRU -> Dense` where both words, word shapes and universal parts-of-speech tokens are embedded and concatenated before feeding to a GRU layer. Embedding dimensions are specified in `seq/Options.jl`. Each option is simply a dictionary with key and value pairs.

Data sets are universal dependency treebanks with available training/dev./test split. Specifically

- The Vietnamese Universal Dependency (VUD) treebank is used to train a Vietnamese part-of-speech tagger (`optionsVUD`). 
- The English Web Treebank (EWT) is used to train an English part-of-speech tagger (`optionsEWT`).
- The Bahasa Indonesia Universal Dependency treebank (GSD) is used to train a Bahasa Indonesian part-of-speech tagger (`optionsGSD`).

## Experiment

The resulting files will be saved to subdirectories of `seq/dat/(lang)/pos` where `lang` can be `eng`, `vie` or `ind`, etc. These directories should exist before training.

To train a tagger, run the file `seq/PoSTagger.jl`. Update the options if necessary. Then run the function `train(options)`, where `options` is the selected options for a language as described above. 

## Bahasa Indonesia Accuracy

- Number of training sentences: 4,094 (with length not greater than 40)
- Number of development sentences: 490
- Number of test sentences: 511
- Options: 20 epochs, batch size = 32, shape embedding size = 4, universal part-of-speech embedding size = 16

| wordSize |  hiddenUnits | trainingAcc | devAcc | testAcc | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    |
| 25 | 32 | 0.9932 | 0.9227 | 0.9220 |  |
| 50 | 32 | 0.9995 | 0.9289 | 0.9280 | |
| 80 | 32 | 0.9970 | 0.9338 | 0.9322 | 5,790 (s) FPT |
| 100 | 32 | 0.9980 | 0.9302 | 0.9276 | 7,563 (s) FPT |
| 25 | 64 | 0.9972 | 0.8854 | 0.8778 |  2,350 (s) FPT |
| 50 | 64 | 0.9987 | 0.9137 | 0.9094 | 4,384 (s) FPT |


# Named Entity Recognition

An implementation of GRU models for named entity recognition is provided in `seq/NameTagger.jl`. The general pipeline is the same as the part-of-speech tagging module, that is `EmbeddingWSP -> GRU -> Dense`. The embedding layer projects words, word shapes, and part-of-speech tags and concatenates these embedding vectors before feeding them to the recurrent layer.

Training data sets come from different sources:

- The VLSP 2016 corpus for training Vietnamese named entity tagger (`optionsVLSP2016`).
- The CoNLL 2003 corpus for training English named entity tagger (`optionsCoNLL2003`).

## Experiment

The resulting files will be saved to subdirectories of `seq/dat/(lang)/ner` where `lang` can be `eng`, `vie` or `ind`, etc. These directories should exist before training.

To train a tagger, run the file `seq/NameTagger.jl`. Update the options if necessary. Then run the function `train(options)`, where `options` is the selected options for a language as described above. After training, run the function `eval(options)` to predict all train/dev./test corpus and save the results to corresponding output files. Finally, run the `conlleval` script on each output file to see the corresponding NER performance.

## Vietnamese VLSP-2016 Accuracy

- Number of training sentences: 16,858 (with length not greater than 40)
- Number of development sentences: 2,831
- Number of test sentences: 2,831 (same as the dev. set)
- Options: 20 epochs, batch size = 32, shape embedding size = 4, part-of-speech embedding size = 16

| wordSize |  hiddenUnits | trainingF1 | devF1 | testF1 | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    |
| 25 | 32 | 0.6828 | 0.4915 | 0.4915 | MBP |
| 50 | 32 | ? | ? | ? | ? |

## English CoNLL-2003 Accuracy

- Number of training sentences: 14,987 (with length not greater than 40)
- Number of development sentences: 3,466
- Number of test sentences: 3,684
- Options: 20 epochs, batch size = 32, shape embedding size = 4, part-of-speech embedding size = 25

| wordSize |  hiddenUnits | trainingF1 | devF1 | testF1 | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    | 
| 50 |  64 | ? | ? | ? | ? (s) Jupiter | 
| 50 |  128 | 0.8475 | 0.6711 | 0.5832 | 18,622 (s) Jupiter | 
| 100 |  64 | 0.8338 | 0.6290 | 0.5210 | 36,909 (s) Jupiter | 
| 100 | 128 | 0.8460 | 0.6227 | 0.5051 | 37,604 (s) Jupiter | 

## Bahasa Indonesia-2020 Accuracy

- Number of training sentences: 1,463 (with length not greater than 40)
- Number of development sentences: 366
- Number of test sentences: 508
- Options: 20 epochs, batch size = 32, shape embedding size = 4, part-of-speech embedding size = 25

| wordSize |  hiddenUnits | trainingF1 | devF1 | testF1 | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    | 
|  25 | 64 | 0.7191 | 0.5720 | 0.5619 | 1,038 (s) MBP | 
|  50 | 64 | 0.7637 | 0.5282 | 0.6009 | 1,869 (s) MBP | 
| 100 | 64 | 0.7741 | 0.5662 | 0.5262 | 3,752 (s) MBP | 

# Dependency Parsing

An implementation of arc-eager dependency parsing algorithm is provided in module `tdp`. The transition classifier use a 
MLP with the following pipeline: 

`Embedding(numFeatures, embeddingSize) -> Dense(embeddingSize, hiddenSize, sigmoid) -> Dense(hiddenSize, numLabels)`

The feature embeddings are trained jointly with the overall model. 

Data sets are universal dependency treebanks with available training/dev./test split. Specifically

- The Vietnamese Universal Dependency (VUD) treebank is used to train a Vietnamese part-of-speech tagger (`optionsVUD`). 
- The English Web Treebank (EWT) is used to train an English part-of-speech tagger (`optionsEWT`).
- The Bahasa Indonesia Universal Dependency treebank (GSD) is used to train a Bahasa Indonesian part-of-speech tagger (`optionsGSD`).

## Experiment

To train a model, run the file `tdp/src/Classifier.jl`, then invoke the function `train(options)` with a desired options for a language. The resulting files will be saved to subdirectories of `tdp/dat/(lang)/` where `lang` can be `eng`, `vie` or `ind`, etc. These directories should exist before training.

The training stops when the accuracy on the validation corpus does not increase after 3 consecutive epochs. 

## Bahasa Indonesia-2020 Accuracy

- Number of training sentences: 4,4094 (with length not greater than 40), resulting in 135,155 training samples for transition classification;
- Number of development sentences: 490 (15,757 validation samples)
- Number of test sentences: 511
- Options: batch size = 32 

| embeddingSize |  hiddenUnits | trainingAcc | devAcc | testAcc | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    | 
|  100 | 64 | ? | ? | ? | ? (s) MBP | 
