# nohup bash ./train_subtaskA_w_prompts.sh ../config/config-bert_base.txt > train_subtaskA_w_prompts_results.log 2>&1 &

###################################################################################################################
###                           Noun Phrase                                       
###################################################################################################################
train_data=/data_subtaskA/noun_phrase_w_prompts/raw_train_all_onecol.csv
dev_data=/data_subtaskA/noun_phrase_w_prompts/raw_val_all_onecol.csv
test_data=/data_subtaskA/noun_phrase_w_prompts/raw_test_all_onecol.csv
kg_data=/data_subtaskA/noun_phrase_w_prompts/raw_test_all_onecol.csv
kg_data2=/data_subtaskA/noun_phrase_w_prompts/raw_test_all_onecol.csv

for seed in {0..3}
do
    echo "start training seed ${seed}......"

    python train_model.py -prompt_index 0 -mode train_en_test_en -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                          -g 0 -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
done
###################################################################################################################
###                           Claim                                      
###################################################################################################################
train_data=/data_subtaskA/claim/raw_train_all_onecol.csv
dev_data=/data_subtaskA/claim/raw_val_all_onecol.csv
test_data=/data_subtaskA/claim/raw_test_all_onecol.csv
kg_data=/data_subtaskA/claim/raw_test_all_onecol.csv
kg_data2=/data_subtaskA/claim/raw_test_all_onecol.csv

for seed in {0..3}
do
    echo "start training seed ${seed}......"

    python train_model.py -prompt_index 0 -mode train_en_test_en -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                          -g 0 -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
done
###################################################################################################################
###                           Mixed                                       
###################################################################################################################
train_data=/data_subtaskA/mixed_w_prompts/raw_train_all_onecol.csv
dev_data=/data_subtaskA/mixed_w_prompts/raw_val_all_onecol.csv
test_data=/data_subtaskA/mixed_w_prompts/raw_test_all_onecol.csv
kg_data=/data_subtaskA/mixed_w_prompts/noun_phrase/raw_test_all_onecol.csv
kg_data2=/data_subtaskA/mixed_w_prompts/claim/raw_test_all_onecol.csv

for seed in {0..3}
do
    echo "start training seed ${seed}......"

    python train_model.py -prompt_index 0 -mode train_en_test_en -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                          -g 0 -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
done
