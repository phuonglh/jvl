# TODO

- Add information about emotional words in the AFINN lexicon (mean of embeddings).
- Add textual features to the model, using BERT.

# DONE

0. Baseline Model 

- Develop the first version for EMP (Empathy Prediction) which does not use essay. This model uses only [:gender, :education, :race, :age, :income] as features. Age and income values are nomalized. Gender, education and race values are one-hot transformed. All features are concatenated. This creates a model as follows:
```
    Chain(
        Dense(17, options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
```
With hidden size is 16, the averaged Pearson correlation is 0.21435 (empathy and distress).

1. AFINN Lexicon Integration
-  Finds the intersection of AFINN lexicon with the corpus lexicon. (`spot` function)
-  Add an un-trained embedding layers, the score is now 0.3267.
-  Use multiple inputs and a trained embedding layers, the training score is now 0.4796 (AFINN embedding size = 16, epoch=80), and the dev. is  score is 0.1869.
```
    Chain(
        Join(
            Embedding(length(lexiconAFINN), options[:afinnSize]),
            identity,
        ),
        Dense(17 + options[:afinnSize], options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
```
-  Use afinnSize=32 => train. score = 0.4153, dev. score = 0.2036.
-  Use unique AFINN tokens instead of list of tokens => train. score = 0.4572, dev. score = 0.158
-  Keep only negative-score tokens => train. score = 0.41895, dev. score = 0.2086

2. NRC Lexicon Integration

- The NRC has 10 emotions; each term is associated with one or more emotions. 
- Add bag of one-hot emotion vectors to the model: 
```
    Chain(
        Join(
            Embedding(length(lexiconAFINN), options[:afinnSize]),
            identity,
        ),
        Dense(10 + 17 + options[:afinnSize], options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
```
The scores are now 0.50465 (train.) and 0.22635 (dev.) with hiddenSize=32.
- Increase hidden size to 50, the scores are now 0.57505 and 0.18535 => overfitted.
- Decrease hidden size to 16, the scores are 0.37305 and 0.1827 => underfitted.
- 32 second times: 0.4689 and 0.2223.

3. AFINN Word Sequence Integration

- Add a GRU layer to compute a vector representation of AFINN words, then concatenate them with other info vector by using JoinR.
```  
    Chain(
        JoinR(
            Embedding(length(lexiconAFINN), options[:afinnSize]),            
            identity,
            GRU(options[:afinnSize], options[:recurrentSize])
        ),
        Dense(10 + 17 + options[:recurrentSize], options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
```
There two extra options: maximum sequence length and recurrent size. 
- maxSeqLen = 16, recurrentSize = 16, Pearson scores = (0.2748, 0.1732)
- maxSeqLen = 8, recurrentSize = 16, Pearson scores = (0.29, 0.185). ==> too long number of AFINN words for each essay.
- maxSeqLen = 4, recurrentSize = 16, Pearson scores = (0.3296, 0.22245)
- maxSeqLen = 4, recurrentSize = 32, Pearson scores = (0.3911, 0.1754) ==> overfitting on the training set.
- maxSeqLen = 4, recurrentSize = 8, Pearson scores = (0.354, 0.20615) ==> underfit

4. Emotion Analysis

On the training data:
```
julia> combine(gdf, nrow)
7×2 DataFrame
 Row │ emotion   nrow  
     │ String15  Int64 
─────┼─────────────────
   1 │ sadness     647
   2 │ neutral     275
   3 │ fear        194
   4 │ anger       349
   5 │ disgust     149
   6 │ surprise    164
   7 │ joy          82
```
One third of the samples are of emotion `sadness`. Only 4% of samples are of emotion `joy`.    