Data and code for "Re-evaluating Evaluation in Text Summarization", an anonymous EMNLP 2020 submission.

## Human Judgement - Annotation Results
The directory data/annotations contains results from crowdsourced annotations (after removing noisy annotators) on summaries from 25 systems x 100 sampled documents from the CNN/DailyMail test set. Each row in the file pertains to one summary, and contains the SCUs (Semantic Content Units) extracted for that summary and the annotation (Present/Not Present) for each SCU. 

## Scoring Data
Scoring data (automated metric scores as well as human (LitePyramids) scores) for system summaries on the 100 sample ddocuments are contained in the file _data/score_dicts/{abstractive,extractive}\_summaries_score_dict.pkl_ 
The dict contains reference summaries, system summaries and system summary scores on multiple metrics.
It can be loaded in python3 as:
```
    import pickle
    with open('<file path>', 'rb') as fp:
        sd = pickle.load(fp)
```
Note that all scores in the file are presented as a fraction of 1, and not as a percentage as typically depicted in literature.

## Code for Analysis
_src/reevaluating_summarization_analysis.ipynb_ contains the code used to analyze the score dict and obtain results for the experiments described in the paper.

## AMT Instructions
_data/amt_template/_ contains the instructions and HTML template used for system summary evaluation on Amazon Mechanical Turk (AMT). This is based on the SCU testing template provided by [LitePyramids](https://github.com/OriShapira/LitePyramids) (Ori Shapira, David Gabay, Yang Gao, Hadar Ronen, Ramakanth Pasunuru, Mohit Bansal, Yael Amsterdamer, and Ido Dagan. Crowdsourcing Lightweight Pyramids for Manual Summary Evaluation. In NAACL 2019.)
