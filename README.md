# Yahoo! Answers Topic Classification

Group project for the Natural Language Processing course at İstinye University.

The project classifies Yahoo! Answers questions and answers into 10 topics. It compares traditional text classifiers with fine-tuned BERT, RoBERTa and DeBERTa-v3 models.

**Repository status:** Code, datasets and individual contribution files are being added. The instructions below apply once the Python script is uploaded.

## Task and dataset

The training code downloads `community-datasets/yahoo_answers_topics` from Hugging Face:

https://huggingface.co/datasets/community-datasets/yahoo_answers_topics

Each original example contains a question title, question content, best answer and topic label. The code combines the three text fields into one input.

The 10 classes are:

- Society & Culture
- Science & Mathematics
- Health
- Education & Reference
- Computers & Internet
- Sports
- Business & Finance
- Entertainment & Music
- Family & Relationships
- Politics & Government

The original dataset has 1.4 million training examples and 60,000 test examples. The updated benchmark shared by the team reports 90,000 training examples per method. Exact prepared split sizes and cleaning statistics should be checked in `data/clean/stats.json`.

## Preprocessing

The latest script:

1. Downloads the source dataset and saves the selected raw examples.
2. Decodes HTML entities, removes HTML tags and normalizes whitespace.
3. Combines the question title, question content and best answer.
4. Removes empty texts and duplicate texts within each source split.
5. Separates validation examples from the training data while approximately preserving class proportions.
6. Saves the cleaned training, validation and test sets.
7. Records class counts, text-length statistics and training/test text overlap.

Overlap is reported for inspection; the script does not automatically remove overlapping examples across training and test sets.

## Files and generated outputs

| Path | Purpose |
|---|---|
| `yahoo_topics_finetune.py` | Preprocessing, baseline training, transformer training, evaluation and benchmark reporting |
| `yahoo_topics_finetune.ipynb` | Colab notebook version of the pipeline |
| `data/raw/` | Selected source examples before cleaning |
| `data/clean/` | Prepared datasets and `stats.json` |
| `results/` | Generated evaluation files and benchmark reports |
| `runs/` | Locally generated model checkpoints |
| `contributions/` | Each member's contribution description |
| `.gitignore` | Rules for excluding unnecessary local files |

The data and results folders are generated when the relevant pipeline stages run. Saved checkpoints are not required for reading the benchmark report.

## Running the project

Use a Python environment with the following packages. A CUDA GPU is recommended for transformer training.

```bash
python -m pip install torch numpy tqdm "transformers>=4.57" "datasets>=4.0" sentencepiece scikit-learn
```

Check the built-in utility tests:

```bash
python yahoo_topics_finetune.py selftest
```

Download and preprocess the data:

```bash
python yahoo_topics_finetune.py prepare
```

Train the traditional baselines:

```bash
python yahoo_topics_finetune.py baselines
```

Fine-tune a single transformer:

```bash
python yahoo_topics_finetune.py train --model roberta-base --seed 42
```

Run the benchmark across BERT, RoBERTa and DeBERTa-v3 with seeds 42, 1 and 2:

```bash
python yahoo_topics_finetune.py benchmark
```

If the individual result files already exist, rebuild the report without training:

```bash
python yahoo_topics_finetune.py benchmark --report_only
```

Benchmark outputs are written to `results/benchmark.md` and `results/benchmark.json`.

## Evaluation

Accuracy measures the proportion of correctly classified examples. It is the headline metric because the original test set is balanced across the 10 classes and it supports comparison with published accuracy scores.

Macro-F1 gives every class equal weight and helps reveal differences in performance between topics. The preprocessing report records the class balance after cleaning.

Transformer checkpoints are selected using validation accuracy and then evaluated on the test set. Transformer inputs are truncated to 128 tokens in the reported runs.

## Team-reported benchmark results

These figures were shared by the team on 11 October 2026. They have not been independently reproduced during repository preparation.

| Method | Accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| RoBERTa-base | 75.89 ± 0.09 | 75.53 ± 0.11 |
| BERT-base-uncased | 75.56 ± 0.06 | 75.19 ± 0.06 |
| Character TF-IDF + linear SVM | 70.14 | 69.69 |
| Word TF-IDF + linear SVM | 69.94 | 69.51 |
| TF-IDF + Complement Naive Bayes | 69.23 | 68.19 |
| TF-IDF + logistic regression | 68.66 | 68.47 |
| DeBERTa-v3-base | 10.00 | 1.82 |
| Majority-class baseline | 10.00 | 1.82 |

BERT, RoBERTa and DeBERTa-v3 results summarize three random seeds. The ± values are sample standard deviations. Traditional baselines use one seed.

The DeBERTa-v3 run did not improve on the majority-class baseline and needs investigation. It should not be interpreted as representative of the model's general performance.

Published reference scores in the generated benchmark use different training budgets and input lengths, so they are contextual comparisons rather than controlled comparisons with our runs.

## Contributions

Each member must provide a description of their actual work in:

`contributions/STUDENT_NAME_STUDENT_ID_GITHUB_USERNAME.md`

## Licensing

Code licensing and dataset redistribution terms are pending confirmation. The dataset's usage terms must be documented separately from the team's code license.
