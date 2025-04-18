# Dutch-RoBERTa-for-Rectal-Cancer-Reports
The Role of Pretrained Language Models in predicting survival

Model
RoBERTa

How to use
To pretrain: check run.sh
To extract features embeddings from pre-trained RoBERTa, use feature_extractor.py and the features will be saved as .npy files
To train survival model: python survival/train.py --feature_dir --output_path  
