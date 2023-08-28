# EMNLP23_code
This is the sample code for our EMNLP23 submission: EZ-STANCE: A Large Dataset for Zero-Shot Stance Detection

cd ./src

To run code for subtask A:
bash ./train_subtaskA.sh ../config/config-MODEL_NAME.txt

To run code for subtask A with prompts:
bash ./train_subtaskA_w_prompts.sh ../config/config-MODEL_NAME.txt

To run code for subtask B:
bash ./train_subtaskB.sh ../config/config-MODEL_NAME.txt

To run code for subtask B with prompts:
bash ./train_subtaskB_w_prompts.sh ../config/config-MODEL_NAME.txt

MODEL_NAME includes:
"bart_mnli_encoder": corresponds to the BART-MNLI-e model
"bert_base": corresponds to the BERT model
"bert_mnli": corresponds to the BERT-MNLI model
"roberta_base_mnli": corresponds to the RoBERTa-MNLI model
"roberta_base": corresponds to the RoBERTa model
"xlnet_base": corresponds to the XLNet model
