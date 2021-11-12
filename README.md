# Introduction

This software package implements some fundamental NLP tasks in the Julia programming language. Tasks which have been implemented so far include:

- intent detection
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

The resulting files will be saved to paths of `dat/pos/$lang-` where `lang` can be `eng`, `vie` or `ind`, etc. 

To train a tagger, run the file `seq/PoSTagger.jl`. Update the options if necessary. Then run the function `train(options)`, where `options` is the selected options for a language as described above. 

## Vietnamese Accuracy (VLSP-2010 treebank)

- Number of training sentences: 8,132 (random split with ratios [0.8, 0.1, 0.1])
- Number of development sentences: 1,016
- Number of test sentences: 1,017
- Options: 20 epochs, batch size = 32, shape embedding size = 4, universal part-of-speech embedding size = 1

| wordSize |  hiddenUnits | trainingAcc | devAcc | testAcc | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    |
| 50  | 64  | 0.9016 | 0.8700 | 0.8728 | 2,971 (s) Jupiter |
| 100 | 64  | 0.9000 | 0.8658 | 0.8697 | 5,854 (s) Jupiter | 
| 100 | 128 | 0.9053 | 0.8586 | 0.8625 | 6,078 (s) Jupiter |


## Vietnamese Accuracy (UD treebank)

- Number of training sentences: 1,400
- Number of development sentences: 800
- Number of test sentences: 800
- Options: 20 epochs, batch size = 32, shape embedding size = 4, universal part-of-speech embedding size = 8

| wordSize |  hiddenUnits | trainingAcc | devAcc | testAcc | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    |
| 16 | 64 | 0.9712 | 0.9669 | 0.9592 | 542 (s) MBP |
| 25 | 64 | 0.9764 | 0.9652 | 0.9555 | 676 (s) MBP |
| 50 | 64 | 0.9797 | 0.9700 | 0.9573 | 1,127 (s) MPB |

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
- The KIK 2020 corpus for training Bahasa Indonesia named entity tagger (`optionsKIK2020`).

## Experiment

The resulting files will be saved to paths of `dat/ner/$lang-` where `lang` can be `eng`, `vie` or `ind`, etc. 

To train a tagger, run the file `seq/NameTagger.jl`. Update the options if necessary. Then run the function `train(options)`, where `options` is the selected options for a language as described above. After training, run the function `evaluate(encoder, options)` to predict all train/dev./test corpus and save the results to corresponding output files. Finally, run the `conlleval` script on each output file to see the corresponding NER performance.

```
    cd jvl/VLP
    julia
    activate .
    include("src/seq/NameTagger.jl")
    options = NameTagger.options...
    options[:wordSize] = 100
    options[:hiddenUnits] = 128
    encoder = NameTagger.train(options)
    NameTagger.evaluate(encoder, options)
```

## Vietnamese VLSP-2016 Accuracy

- Number of training sentences: 16,858 (with length not greater than 40)
- Number of development sentences: 2,831
- Number of test sentences: 2,831 (same as the dev. set)
- Options: 20 epochs, batch size = 32, shape embedding size = 4, part-of-speech embedding size = 16

| wordSize |  hiddenUnits | trainingF1 | devF1 | testF1 | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    |
| 25  | 16 | 0.6654 | 0.4742 | 0.4742 | 11,557 (s) Jupiter | 
| 25  | 32 | 0.6828 | 0.4915 | 0.4915 | ? MBP |
| 25  | 64 | 0.6786 | 0.4480 | 0.4480 | 12,530 (s) Jupiter |
| 50  | 32 | 0.6730 | 0.4942 | 0.4942 | 14,616 (s) Jupiter |
| 100 | 16 | 0.6694 | 0.4829 | 0.4829 | 48,858 (s) Jupiter | 
| 100 | 32 | 0.6560 | 0.4864 | 0.4864 | 30,008 (s) Jupiter |
| 100 | 64 | 0.6637 | 0.4554 | 0.4554 | 30,618 (s) Jupiter |
| 100 | 128| 0.7090 | 0.4343 | 0.4343 | 40,645 (s) Jupiter |

## English CoNLL-2003 Accuracy

- Number of training sentences: 14,987 (with length not greater than 40)
- Number of development sentences: 3,466
- Number of test sentences: 3,684
- Options: 20 epochs, batch size = 32, shape embedding size = 4, part-of-speech embedding size = 25

| wordSize |  hiddenUnits | trainingF1 | devF1 | testF1 | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    | 
| 50 |   64 | 0.8333 | 0.6128 | 0.5072 | 17,728 (s) Jupiter | 
| 50 |  128 | 0.8475 | 0.6711 | 0.5832 | 18,622 (s) Jupiter | 
| 100 |  64 | 0.8338 | 0.6290 | 0.5210 | 36,909 (s) Jupiter | 
| 100 | 128 | 0.8460 | 0.6227 | 0.5051 | 37,604 (s) Jupiter | 
| 100 | 256 | 0.8270 | 0.6225 | 0.4984 | 39,825 (s) Jupiter | 

## Bahasa Indonesia-2020 Accuracy

- Number of training sentences: 1,463 (with length not greater than 40)
- Number of development sentences: 366
- Number of test sentences: 508
- Options: 20 epochs, batch size = 32, shape embedding size = 4, part-of-speech embedding size = 16

| wordSize |  hiddenUnits | trainingF1 | devF1 | testF1 | trainingTime
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    | 
|  25 | 64 | 0.7191 | 0.5720 | 0.5619 | 1,038 (s) MBP | 
|  50 | 64 | 0.7637 | 0.5282 | 0.6009 | 1,869 (s) MBP | 
| 100 | 64 | 0.7741 | 0.5662 | 0.5262 | 3,752 (s) MBP | 
| 100 | 128| 0.8044 | 0.5243 | 0.5049 | 1,131 (s) Jupiter |

# Dependency Parsing

An implementation of arc-eager dependency parsing algorithm is provided in module `tdp`. The transition classifier use a 
MLP with the following pipeline: 

`Embedding(numFeatures, embeddingSize) -> Dense(embeddingSize, hiddenSize, sigmoid) -> Dense(hiddenSize, numLabels)`

The feature embeddings are trained jointly with the overall model. There are two flavors of the model:

- The CBOW model with `Embedding` layer
- The concatenation model with `EmbeddingConcat` layer

Data sets are universal dependency treebanks with available training/dev./test split. Specifically

- The Vietnamese Universal Dependency (VUD) treebank is used to train a Vietnamese part-of-speech tagger (`optionsVUD`). 
- The English Web Treebank (EWT) is used to train an English part-of-speech tagger (`optionsEWT`).
- The Bahasa Indonesia Universal Dependency treebank (GSD) is used to train a Bahasa Indonesian part-of-speech tagger (`optionsGSD`).

## Experiment

To train a model, run the file `tdp/Classifier.jl`, then invoke the function `train(options)` with a desired options for a language. The resulting files will be saved to subdirectories of `dat/tdp/$lang-` where `lang` can be `eng`, `vie` or `ind`, etc. 

The training stops when the accuracy on the validation corpus does not increase after 3 consecutive epochs. 

The `tdp/Oracle.jl` utility extracts features from parsing configurations. Each parsing config has an associated stack, buffer and partial arc list. Two top tokens on the stack, two top tokens on the buffer are considered; each has 5 features. Each parsing configuration has thus 20 feature strings. 

```
    cd jvl/VLP
    julia
    include("src/tdp/Classifier.jl")
    options = TransitionClassifier.optionsVUD
    # change parameters before training:
    options[:embeddingSize] = 100
    options[:hiddenSize] = 64
    mlp = TransitionClassifier.train(options)
    # load graphs for evaluating:
    using Flux
    mlp, featureIndex, labelIndex = TransitionClassifier.load(options)
    sentences = TransitionClassifier.readCorpusUD(options[:testCorpus]);
    TransitionClassifier.evaluate(mlp, featureIndex, labelIndex, options, sentences)
```

After training a classifier, invoke the parser to parse or evaluate its accuracy on sets of sentences.

- `run(options, sentences)`: loads a classifier and run the parser on `sentences`, the given sentences are updated directly where each token has `:h` and `:l` annotation specifying its head and dependency label
- `evaluate(options, sentences)`: evaluates parsing performance in terms of UAS (unlabeled attachment score) and LAS (labeled attachment score). 
- `evaluate(options)`: loads training/dev./test data sets from the given `options` and evaluates parsing performance on these sets.

```
    cd jvl/VLP
    julia
    include("src/tdp/Parser.jl")
    options = DependencyParser.TransitionClassifier.optionsVUD
    using Flux
    mlp, featureIndex, labelIndex = TransitionClassifier.load(options)
    sentences = DependencyParser.readCorpusUD(options[:testCorpus]);
    # DependencyParser.evaluate(mlp, featureIndex, labelIndex, options, sentences)
    DependencyParser.evaluate(options)
```


## Bahasa Indonesia Accuracy

- Number of training sentences: 4,4094 (with length not greater than 40), resulting in 135,155 training samples for transition classification;
- Number of development sentences: 490 (15,757 validation samples)
- Number of test sentences: 511
- Number of depdendency labels: 
- Options: batch size = 32 

| embeddingSize |  hiddenUnits | trainingAcc | devAcc | testAcc | trainingTime | epochs | trainUAS | trainLAS | devUAS | devLAS | testUAS | testLAS | 
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 50 |  32 | 0.8224 | 0.6039 | ? | 1,701 (s) T480s | 8 | 0.5932 | 0.5432 | 0.4514 | 0.3822 | 0.4476 | 0.8331 |
| 50 | 32 |  0.7952 | 0.6005 | ? | 1,493 (s) T480s | 7 |
| 50 |  64 | 0.8176 | 0.6120 | ? | 1,460 (s) T480s | 7 |
| 50 | 64 |  0.8099 | 0.6215 | ? | 1,277 (s) T480s | 2 |
| 50 | 128 | 0.8010 | 0.6039 | ? | 1,078 (s) T480s | 7 |
| 50 | 128 | 0.8169 | 0.6273 | ? | 820 (s) T480s  | 2 |
| 100 | 128 | 0.7601 | 0.5851 | ? | 3,591 (s) T480s | 6 |


## Vietnamese Accuracy

- Number of training sentences: 1,400 (35,888 contexts)
- Number of development sentences: 800 (20,321 contexts)
- Number of test sentences: 800 (21,161)
- Number of dependency labels: 52

| embeddingSize |  hiddenUnits | trainingAcc | devAcc | testAcc | trainingTime | epochs | trainUAS | trainLAS | devUAS | devLAS | testUAS | testLAS |
| ---:       | :---:   | :---:    | :---:    | :---:    | :---:    | :---:    |   :---: | :---: |  :---: | :---: |  :---: | :---: |
|  50 | 32 | 0.8118 | 0.5790 | 0.5652 | 312 (s) MBP | 10 | 0.5416 | 0.5154 | 0.3379 | 0.2804 | 0.3203 | 0.2595 | 
|  50 | 64 | 0.8100 | 0.5703 | 0.5527 | 164 (s) T480s | 9 |
|  50 | 128 | 0.8552 | 0.5820 | 0.5656 | 478 (s) MBP | 10 | 0.6473 | 0.5987 | 0.4099 | 0.3252 | 0.3929 | 0.3037 | 
| 100 | 32  | 0.6120 | 0.4734 | 0.4606 | 952 (s) MBP | 11 | 0.3433 | 0.2163 | 0.2596 | 0.1515 | 0.2483 | 0.1336 | 
| 100 | 64 | 0.7612 | 0.5480 | 0.5276 | 899 (s) MBP | 10 | 0.5821 | 0.4749 | 0.4078 | 0.2963 | 0.3853 | 0.2707 | 
| 100 | 128 | 0.8309 | 0.5685 | 0.5514 | 1,204 (s) MBP | 13 | 0.6409 | 0.5907 | 0.3932 | 0.3127 | 0.3706 | 0.2851 | 
| 100 | 256 | 0.7419 | 0.5492 | 0.5371 | 929 (s) MBP | 10 | 0.4135 | 0.3861 | 0.2699 | 0.2211 | 0.2627 | 0.2093 | 

# Arc-Eager Parsing (AEP)

Train and evaluate the transition classifier:

```
    cd jvl/VLP
    julia
    include("src/aep/Classifier.jl")
    options = TransitionClassifier.optionsVUD
    # change parameters before training:
    options[:hiddenSize] = 64
    mlp = TransitionClassifier.train(options)
    # load graphs for evaluating:
    sentences = TransitionClassifier.readCorpusUD(options[:testCorpus]);
    using Flux
    TransitionClassifier.evaluate(options, sentences)
```

Train and evaluate the parser:

```
    cd jvl/VLP
    julia
    include("src/aep/Parser.jl")
    options = ArcEagerParser.TransitionClassifier.optionsVUD
    using Flux
    ArcEagerParser.evaluate(options)
```

Perform experiments:

```
    cd jv/VLP
    julia
    include("src/aep/Experiment.jl")
    options = ArcEagerParser.TransitionClassifier.optionsVUD 
    # update options
    options[:embeddingSize] = 64
    options[:hiddenSize] = 128
    experiments(options)
    # open the result file `/dat/aep/$lang-score.jsonl`
```

# Graph Embedding

To train graph embeddings for a language:

- Open `src/emb/TransE.jl`, edit the `language` selection line.
- Run this file.
- The output should be saved in `dat/emb/`.

# Extended Graph Embedding Features for AEP

Train and evaluate the extended transition classifier:

```
    cd jvl/VLP
    julia
    include("src/aep/ClassifierEx.jl")
    options = TransitionClassifierEx.optionsVUD
    # change parameters before training:
    options[:hiddenSize] = 64
    mlp = TransitionClassifierEx.train(options)
    # load graphs for evaluating:
    sentences = TransitionClassifierEx.readCorpusUD(options[:testCorpus]);
    using Flux
    TransitionClassifierEx.evaluate(options, sentences)
```

Train and evaluate the extended parser:

```
    cd jvl/VLP
    julia
    include("src/aep/ClassifierEx.jl")
    include("src/aep/ParserEx.jl")
    options = ArcEagerParserEx.TransitionClassifierEx.optionsVUD
    using Flux
    ArcEagerParserEx.evaluate(options)
```
# Vietnamese Spelling Check 

The `src/vsc/Kar.jl` code implements a semi-character RNN (GRU) model that predict 
the jumbling type of a Vietnamese syllable. There are five types: swap (`:s`), delete (`:d`), replace (`:r`), 
insert (`:i`) and none (`:n`). 

Given a sentence, for each of its syllable, we randomly apply a jumbling 
method on the syllable with a probablity p, for example p = 0.1, to make it a spelling error. 
The entire sentence is then used as an example to train a sequence prediction model, where 
an input is a sequence of syllables `[s_1, s_2,...,s_N]` and its corresponding output is a jumbling 
sequence `[y_1, y_2,...,y_N]` where a label `y_k` in the set `[:s, :d, :r, :i, :n]`.

To train a `Kar` model with some options, simply run `Kar.train(Kar.options)`. 
