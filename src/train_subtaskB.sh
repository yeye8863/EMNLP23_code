# nohup bash ./train_subtaskB.sh ../config/config-bert_base.txt > train_subtaskB_results.log 2>&1 &

###################################################################################################################
###                           Noun Phrase                                       
###################################################################################################################
for domain in "covid19_domain" "world_event_domain" "education_and_culture_domain" "consumption_and_entertainment_domain" "sports_domain" "rights_domain" "environmental_protection_domain" "politic"
do
  echo "start training on domain ${domain}......"
  train_data=/data_subtaskB/noun_phrase/${domain}/raw_train_all_onecol.csv
  dev_data=/data_subtaskB/noun_phrase/${domain}/raw_val_all_onecol.csv
  test_data=/data_subtaskB/noun_phrase/${domain}/raw_test_all_onecol.csv
  kg_data=/data_subtaskB/noun_phrase/${domain}/raw_test_all_onecol.csv
  kg_data2=/data_subtaskB/noun_phrase/${domain}/raw_test_all_onecol.csv

  for seed in {0..3}
  do
      echo "start training seed ${seed}......"

      python train_model.py -prompt_index 0 -mode train_en_test_en -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                            -g 0 -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
  done  
done

###################################################################################################################
###                           Claim                                      
###################################################################################################################
for domain in "covid19_domain" "world_event_domain" "education_and_culture_domain" "consumption_and_entertainment_domain" "sports_domain" "rights_domain" "environmental_protection_domain" "politic"
do
  echo "start training on domain ${domain}......"
  train_data=/data_subtaskB/claim/${domain}/raw_train_all_onecol.csv
  dev_data=/data_subtaskB/claim/${domain}/raw_val_all_onecol.csv
  test_data=/data_subtaskB/claim/${domain}/raw_test_all_onecol.csv
  kg_data=/data_subtaskB/claim/${domain}/raw_test_all_onecol.csv
  kg_data2=/data_subtaskB/claim/${domain}/raw_test_all_onecol.csv

  for seed in {0..3}
  do
      echo "start training seed ${seed}......"

      python train_model.py -prompt_index 0 -mode train_en_test_en -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                            -g 0 -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
  done
done
###################################################################################################################
###                           Mixed                                       
###################################################################################################################
for domain in "covid19_domain" "world_event_domain" "education_and_culture_domain" "consumption_and_entertainment_domain" "sports_domain" "rights_domain" "environmental_protection_domain" "politic"
do
  echo "start training on domain ${domain}......"
  train_data=/data_subtaskB/mixed/${domain}/raw_train_all_onecol.csv
  dev_data=/data_subtaskB/mixed/${domain}/raw_val_all_onecol.csv
  test_data=/data_subtaskB/mixed/${domain}/raw_test_all_onecol.csv
  kg_data=/data_subtaskB/mixed/${domain}/raw_test_all_onecol.csv
  kg_data2=/data_subtaskB/mixed/${domain}/raw_test_all_onecol.csv

  for seed in {0..3}
  do
      echo "start training seed ${seed}......"

      python train_model.py -prompt_index 0 -mode train_en_test_en -c $1 -train ${train_data} -dev ${dev_data} -test ${test_data} -kg ${kg_data} -kg2 ${kg_data2} \
                            -g 0 -s ${seed} -d 0.1 -clipgrad True -step 3  --earlystopping_step 5 -p 100
  done
done