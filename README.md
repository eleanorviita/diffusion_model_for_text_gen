# Diffusion Models for Text Generation - MSc Project

Two main types of machine learning models are used in natural language processing (NLP): autoregressive (AR) models, where text is generated in sequence, extrapolating each word (or token) in a sentence from the previous one, and non-autoregressive (NAR) models, which generate multiple tokens in parallel. Diffusion models are a relatively recent development in NLP. They work by iteratively refining noise from a given input, until a coherent output is produced. These models have been used successfully in image generation and are now being applied to generate text.

This project set out to implement diffusion techniques for text generation and evaluate the performance and effectiveness of diffusion models compared to AR and NAR baselines, based on a selection of typical tasks and benchmarks in literature.  

This repository contains code for three model types across two text generation tasks:
- Controllable generation (i.e. text written in a particular style, sentiment or formality)
- Open-ended generation (e.g. continuing the text "once upon a time...")

### Model types:
- Autoregressive (AR, GPT-2 fine-tuned)
- Non-autoregressive (NAR, BERT mask-predict, with and without knowledge distillation)
- Diffusion (T5 encoder + custom diffusion decoder)

## Repository structure
```
.
├─ models/
│  ├─ trained models (not included here)
│  ├─ ar.py
│  ├─ nar.py
│  ├─ diffusion.py
│  ├─ formality_classifier.py
│  └─ sentiment_classifier.py
├─ data/
│  ├─ preprocess.py
│  ├─ utils.py
│  ├─ GYAFC_Corpus (not included here)
│  └─ yelp_formality_tagged_clean.jsonl (not included here)
├─ tasks/
│  ├─ controllable.py
│  ├─ open_ended.py
│  └─ task_utils.py
├─ workflows/
│  ├─ diffusion_toy_test.py
│  └─ train_model_for_task.py
├─ requirements.txt
└─ README.md
```

## Datasets
The Yelp polarity dataset and Grammarly Yahoo Answers Formality Corpus were used to train the text generation models for fine-grained control of formality and sentiment. A BERT-based classifier was trained on the GYAFC dataset and then used to categorise the Yelp dataset, giving a balanced training dataset with all combinations of formal/informal AND positive/negative.

The WikiText-103 dataset was used as a training dataset for open-ended text generation.


## Set-up
A requirements.txt file is saved in the repository with a list of all required Python libraries, and can be used to install dependencies:  

```
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Running training scripts

There are pre-written function calls at the end of train_model_for_task.py which can be uncommented (and the training arguments edited if desired) to run the relevant model training. Once the relevant line(s) are uncommented, run the training script as a module:

```
python -m workflows.train_model_for_task
```

This section also includes the optional generate_teacher_outputs(...) function which uses the trained AR model to generate teacher examples for knowledge distillation. These can then be used for NAR knowledge distilled training.

Model names are automatically generated as  
<model_name> = {base_model}\_{model_type}\_{task}  
Checkpoints and final trained models are saved under models / <model_name> /

## Formality tagging
The BERT formality and sentiment classifiers are used by controllable.py to evaluate prompt adherence. The formality_classifier.py and sentiment_classifier.py scripts can also be run on their own as a module to train a new classifier.  

```
python -m models.formality_classifier
python -m models.sentiment_classifier
```
formality_classifier.py will also classify the original Yelp dataset into formal/informal examples and create a new formality-labelled dataset for training.  
A pre-made formality-labelled Yelp dataset is saved as data/yelp_formality_tagged_clean.jsonl.

## Running inference scripts

To run controllable generation and evaluation for all trained models:  
```
python -m tasks.controllable
```

This will generate controllable text using all trained models: AR, NAR, NAR (knowledge distilled) and Diffusion, evaluate perplexity and sentiment/formality accuracy and save the results under tasks/controllable_results/  

Likewise for open-ended generation:  
```
python -m tasks.open_ended
```

This will generate open-ended text using the all trained models: AR, NAR, NAR (knowledge distilled) and Diffusion, evaluate perplexity and diversity metrics (d1, d2, self-BLEU) and save the results under tasks/open_ended_results/  



## Acknowledgements
HuggingFace Transformers and Datasets  
Wikitext-103  
Yelp polarity
Grammarly Yahoo Answers Formality Corpus

## License
Academic / research use
