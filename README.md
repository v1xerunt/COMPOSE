# COMPOSE: Cross-Modal Pseudo-Siamese Network for Patient Trial Matching

The source code for KDD 2020 paper *COMPOSE: Cross-Modal Pseudo-Siamese Network for Patient Trial Matching*

Oral presentation video: https://youtu.be/OHEjb1IhoMU
Slides: http://aboutme.vixerunt.org/publication/compose/COMPOSE_slides.pdf

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* If you plan to use GPU computation, install CUDA
* Install the python package pytorch-pretrained-bert. 
* Download the pre-trained clinical BERT (https://github.com/EmilyAlsentzer/clinicalBERT) weight from 
https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz?dl=1

## Data preparation

* EHR Data: EHR data should be saved in ```./ehr_data/``` directory. EHR data consists of four files - diagnosis_list.csv, procedure_list.csv, product_list.csv, records.csv. 
    * The diagnosis/procedure/product tables save the entire medical codes set and the corresponding hiearchical textual description informations (lv1-lv4) for each code. We use the Uniform-System-of-Classification taxonomy (https://www.cdc.gov/antibiotic-use/community/pdfs/Uniform-System-of-Classification-2018-p.pdf)
    * The records.csv saves patients' visits. Each visit should have diagnosis/procedure/product codes. And this table should also stores the information about the corresponding trials for each patient.
    * We provide sample EHR data and codes in ```./ehr_data/``` directory.
* Trial data: Trial data should be saved in ```./trial_data/``` directory. You can find certain trials in https://clinicaltrials.gov/. Or you can download multiple trials from https://clinicaltrials.gov/ct2/resources/download. The downloaded trials are xml files.
    * We also provide the trial data used in our experiment. Please use ```pickle.load('./trial_list', 'rb')``` to load trial data.

## Data pre-processing
1. We first use ```./utils/filter_trial.py``` to load all trial xml files to a list.
2. Then use ```./utils/ECpreprocess.py``` to extract inclusion and exclusion criteria, and segment text blocks to lists of criteria.
3. We use ```./utils/generate_embeddings.py``` to use pretrained BERT to generate word embeddings for trial criteria.
4. We use ```./utils/code_embedding.py``` to generate word embeddings for each medical code and its parent codes using BERT.
5. Finally we use ```./utils/generate_ehr_dataset.py``` to generate EHR dataset for training.

## Training COMPOSE

The model structure for COMPOSE is defined in ```./model.py```. You can use ```./train_model.py``` to train COMPOSE model. Hyper-parameters such as learning rate or dimension of memory network can be changed in ```train_model.py``` file.

Input data description:
1. EHR data: (batch_size, time_step, 12, word_dim)
EHR data consists of three codes: diagnosis, medication and procedure. Each codes is represented by 4-level text description. So at each visit, the patient's ehr data is a 12*word_dim matrix.
2. EHR mask: (batch_size, time_step) 1/0 mask to indicate whether the time_step is a padding.
3. EHR demographic: (batch_size, 3) Demographic data of each patient including age and gender.
4. Trial criteria: (batch_size, word_len, word_dim) A criteria sample
5. Criteria mask: (batch_size, word_len): Indicate padding value.
6. Label: (batch_size): 0 for mismatch, 1 for match, 2 for unknown. We randomly sampled a criterion as unknown(2) for each known criterion (0/1).

## Testing COMPOSE

The trained weights will be saved in ```./save/``` folder. After training, you can use ```./test_model.py``` to evaluate COMPOSE. The code will output AUROC, AUPRC, Accuracy and F1 scores. 

## Citation
```
Junyi Gao, Cao Xiao, Lucas M. Glass, and Jimeng Sun. 2020. 
COMPOSE: Cross-Modal Pseudo-Siamese Network for Patient Trial Matching. 
In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20), August 23–27, 2020, Virtual Event, CA, USA. ACM, New York, NY, USA, 10 pages. 
https://doi.org/10.1145/3394486.3403123
```
