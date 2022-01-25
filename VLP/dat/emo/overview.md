# WASSA 2022 Shared Task on Empathy Detection and Emotion Classification

Emotion is a concept that is challenging to describe. Yet, as human beings, we understand the emotional effect situations have or could have on us and other people. How can we transfer this knowledge to machines? Is it possible to learn the link between situations and the emotions they trigger in an automatic way?

We propose the Shared Task on Empathy Detection, Emotion Classification and Personnality Detection, organized as part of [WASSA 2022](https://wassa-workshop.github.io/) at [ACL 2022](https://www.2022.aclweb.org/). This task aims at developing models which can predict empathy and emotion based on essays written in reaction to news articles where there is harm to a person, group, or other.

## Task Description

Participants are given an extended release of the empathic reactions to news stories dataset which contains essays and Batson empathic concern and personal distress scores in reaction to news articles where there is harm to a person, group, or other (for more details see [Buechel et al. 2018](https://www.aclweb.org/anthology/D18-1507/)). The essays are between 300 and 800 characters in length. The extension of this dataset also includes the news articles as well as person-level demographic information (age, gender, ethnicity, income, education level).

You can participate in four different tracks:

- **Track 1: Empathy Prediction (EMP)**, which consists in predicting both the empathy concern and the personal distress at the essay-level
- **Track 2: Emotion Classification (EMO)**, which consists in predicting the emotion at the essay-level
- **Track 3: Personnality Prediction (PER)**, which consists in predicting the personnality of the essay writer, knowing all his/her essays and the news article from which they reacted
- **Track 4: Interpersonal Reactivity Index Prediction (IRI)**, which consists in predicting the personnality of the essay writer, knowing all his/her essays and the news article from which they reacted

Below is an example of essays and labels.

| Empathy|Distress|Essay|Age|Income|Emotion|Personality_Openess|
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
4.8 |3.1|it is really diheartening to read about these immigrants from this article who drowned.  it makes me feel anxious and upset how the whole ordeal happened.  it is a terrible occurrence that this had to happen at the mediterranean sea.  thankfully there were some survivors.  the fact that babies were lost makes it that much more emotional to read all of this|33|5000|sadness|5|
3.3|3.5|I think almost everyone has an opinion on Hilary Clinton over Donald Trump. Many didn't expect Donald to win the election but here we are. So many political scandals unfolded and more continuing to unravel. Many didn't agree with Hilary from the Bengazi investigation and many didn't agree with Trump, but in the end that's just politics.|25|7500|anger|7
|

## Paper

Participants will be given the opportunity to write a system-description paper that describes their system, resources used, results, and analysis. This paper will be part of the official WASSA-2022 proceedings. The paper is to be four pages long plus two pages at most for references and should be submitted using the ACL 2022 Style Files ([LaTeX style files](https://github.com/acl-org/acl-style-files)) on [ACL Rolling Review](https://aclrollingreview.org/cfp). The paper can contain an appendix.

## Evaluation
For development purposes, we provide an evaluation script here (`evaluation.py`). The script takes two or three files as input, a gold-standard file (such as the gold standard of the train) and one or two prediction files in the format described in 'Submission Format'.

### Track 1 (EMP):

**Official Competition Metric**: the evaluation will be based on **the average of the two Pearson correlations** below:
- Pearson correlation of the empathy
- Pearson correlation of the distress

### Track 2 (EMO):

**Official Competition Metric**: the evaluation will be based on the macro F1-score. **Secondary Evaluation Metrics**: Apart from the official competition metric described above, some additional metrics will also be calculated for your submissions. These are intended to provide a different perspective on the results:

- Accuracy
- Micro F1-score
- Micro Precision
- Micro Recall
- Macro Precision
- Macro Recall

### Track 3 (PER):

**Official Competition Metric**: the evaluation will be based on the average of **the Pearson correlations over Personnality values**: Conscientiousness, Openess, Extraversion, Agreeableness and Stability

### Track 4 (IRI):

**Official Competition Metric**: the evaluation will be based on **the average of the Pearson correlations over IRI values**: Perspective-taking, Personal distress, Fantasy and Empathatic concern.

## Datasets

- Training data: `messages_train_ready_for_WS.tsv`
- Development data: `messages_dev_features_ready_for_WS_2022.tsv`

## Schedule

- January 3rd 2022: Initial training data release
- January 3rd 2022: Codalab competition website goes online and development data released
- March 13th 2022: Evaluation phase begins: development labels - test data released
- March 16th 2022: Deadline submission final result on Codalab
- March 21st 2022: Deadline system description paper (max. 4p)
- April 3rd 2022: Notification of acceptance
- April 10th 2022: Camera-ready papers due

## Submission format

System submissions for CodaLab are zip-compressed folders containing one, two or three predictions files called "predictions_EMP.tsv" and "predictions_EMO.tsv" and "predictions_PER.tsv" and "predictions_IRI.tsv".

The evaluation script will check whether the file contains the correct number of instances.

## Training and Evaluation phase

During the training phase (now - March 13th 2022), teams can upload a submission by means of development. They can upload predictions for all instances in the development data in the same way as for the official evaluation phase. During the training phase, submissions will be evaluated against the gold-standard labels of the development data.

Find below a step-by-step guideline to upload your submission on CodaLab during both the development and evaluation phase:

- the output of your system should be saved in files named "predictions_EMP.tsv" and/or "predictions_EMO.tsv" and/or "predictions_PER.tsv" and/or "predictions_IRI.tsv", depending on the tasks you want to compete in. For the EMP subtask, we expect a tsv file in which the first column (tab-separated from everything else) contains the prediction values for empathy, and the second column the prediction values for distress. For the EMO subtask, the first column the prediction values for emotion as a string format. For the PER subtask, the 5 columns are the prediction values for PER in this order: Conscientiousness, Openess, Extraversion, Agreeableness and Stability. For the PER subtask, the 4 columns are the prediction values for IRI in this order: Perspective-taking, Personal distress, Fantasy and Empathatic concern. The samples need to be in the same order as our dev/test set. Everything else than those columns is optional and will be ignored. The predictions start at the first line (no header). If you participate to one of the tasks only, submit only one file.
- compress this file into a zipfile "predictions.zip" (for Mac users: ensure that this zipfile contains no _macosx file).
- on CodaLab, navigate to 'Participate' > Submit/View Results and upload your "predictions.zip" file
- click 'Refresh status' until your submission receives the "Finished" status
- click 'Submit to leaderboard' to push your results to the official scoring board. Please note: as soon as the official evaluation period starts, the scoring board will not be made visible until the official announcements of the final results (March 16th 2022).

