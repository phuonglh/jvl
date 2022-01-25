# TODO

- Add information about emotional words in the AFINN lexicon (mean of embeddings).
- Add textual features to the model, using BERT.

# DONE

- Develop the first version for EMP (Empathy Prediction) which does not use essay. This model uses only [:gender, :education, :race, :age, :income] as features. Age and income values are nomalized. Gender, education and race values are one-hot transformed. All features are concatenated. This creates a model as follows:
```
    Chain(
        Dense(17, options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
```

With hidden size is 16, the averaged Pearson correlation is 0.21435 (empathy and distress).
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
-  Use afinnSize=32 => train. score is 0.4153, dev. score is 0.2036.