import os
os.environ["CUDA_VISIBLE_DEVICES"] ="2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import argparse
import json
import gc
import gspread
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, evaluation, model_utils
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,classification_report

from torch.utils.tensorboard import SummaryWriter   
from pytorchtools import EarlyStopping



# CUDA_VISIBLE_DEVICES=0

def compute_performance(preds,y,trainvaltest,step,args,seed):
    print("preds:",preds,preds.size())
    print("y:",y,y.size())
    preds_np = preds.cpu().numpy()
    # preds_np = preds_np[:,[0,2,1]]
    preds_np = np.argmax(preds_np, axis=1)
    y_train2_np = y.cpu().numpy()
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')

    print("-------------------------------------------------------------------------------------")
    print(trainvaltest + " classification_report for step: {}".format(step))
    target_names = ['Against', 'Favor', 'neutral']
    print(classification_report(y_train2_np, preds_np, target_names = target_names, digits = 4))
    ###############################################################################################
    ################            Precision, recall, F1 to csv                     ##################
    ###############################################################################################
    # y_true = out_label_ids
    # y_pred = preds
    results_twoClass = precision_recall_fscore_support(y_train2_np, preds_np, average=None)
    results_weighted = precision_recall_fscore_support(y_train2_np, preds_np, average='macro')
    print("results_weighted:",results_weighted)
    result_overall = [results_weighted[0],results_weighted[1],results_weighted[2]]
    result_against = [results_twoClass[0][0],results_twoClass[1][0],results_twoClass[2][0]]
    result_favor = [results_twoClass[0][1],results_twoClass[1][1],results_twoClass[2][1]]
    result_neutral = [results_twoClass[0][2],results_twoClass[1][2],results_twoClass[2][2]]

    print("result_overall:",result_overall)
    print("result_favor:",result_favor)
    print("result_against:",result_against)
    print("result_neutral:",result_neutral)

    result_id = ['train', args['gen'], step, seed, args['dropout'],args['dropoutrest']]
    result_one_sample = result_id + result_against + result_favor + result_neutral + result_overall
    result_one_sample = [result_one_sample]
    print("result_one_sample:",result_one_sample)

    # if results_weighted[2]>best_train_f1macro:
    #     best_train_f1macro = results_weighted[2]
    #     best_train_result = result_one_sample

    results_df = pd.DataFrame(result_one_sample)    
    print("results_df are:",results_df.head())
    results_df.to_csv('./results_'+trainvaltest+'_df.csv',index=False, mode='a', header=False)    
    print('./results_'+trainvaltest+'_df.csv save, done!')
    print("----------------------------------------------------------------------------")

    return results_weighted[2],result_one_sample

def run_classifier():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-g', '--gen', help='Generation number of student model', required=False)
    parser.add_argument('-s', '--seed', help='Random seed', required=False)
    parser.add_argument('-d', '--dropout', help='Dropout rate', required=False)
    parser.add_argument('-d2', '--dropoutrest', help='Dropout rate for rest generations', required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file', default=None, required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file', default=None, required=False)
    parser.add_argument('-kg', '--kg_data', help='Name of the kg test data file', default=None, required=False)
    parser.add_argument('-clipgrad', '--clipgradient', type=str, default='True', help='whether clip gradient when over 2', required=False)
    parser.add_argument('-step', '--savestep', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-p', '--percent', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-es_step', '--earlystopping_step', type=int, default=1, help='whether clip gradient when over 2', required=False)
    parser.add_argument('-mode', '--mode', type=str, default='train_en_test_zh', help='language for trainval, and test', required=True)
    parser.add_argument('-prompt_index', '--prompt_index', type=int, default=1, help='index to make nounphrase targets to a sentence', required=True)

    args = vars(parser.parse_args())




    # writer = SummaryWriter('./tensorboard/')

    sheet_num = 4  # Google sheet number
    num_labels = 3  # Favor, Against and None
#     random_seeds = [0,1,2,3,4,42]
    random_seeds = []
    random_seeds.append(int(args['seed']))
    
    # create normalization dictionary for preprocessing
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1,**data2}
    
    # load config file
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    model_select = config['model_select']
    
    # Use GPU or not
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    best_result, best_against, best_favor, best_val, best_val_against, best_val_favor,  = [], [], [], [], [], []
    for seed in random_seeds:    
        print("current random seed: ", seed)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'train')
        train_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'val')
        val_writer = SummaryWriter(log_dir=log_dir)

        log_dir = os.path.join('./tensorboard/tensorboard_train'+str(args['percent'])+'_d0'+str(args['dropout'])+'_d1'+str(args['dropoutrest']+'_seed'+str(seed)+'_gen'+str(args['gen'])), 'test')
        test_writer = SummaryWriter(log_dir=log_dir)

        # set up the random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        x_train, y_train, x_train_target = pp.clean_all(args['train_data'], norm_dict,args['mode'],'train',args['prompt_index'])
        x_val, y_val, x_val_target = pp.clean_all(args['dev_data'], norm_dict,args['mode'],'val',args['prompt_index'])
        x_test, y_test, x_test_target = pp.clean_all(args['test_data'], norm_dict,args['mode'],'test',args['prompt_index'])
        x_test_kg, y_test_kg, x_test_target_kg = pp.clean_all(args['kg_data'], norm_dict,args['mode'],'val',args['prompt_index'])
        x_train_all = [x_train,y_train,x_train_target]
        x_val_all = [x_val,y_val,x_val_target]
        x_test_all = [x_test,y_test,x_test_target]
        x_test_kg_all = [x_test_kg,y_test_kg,x_test_target_kg]
        if int(args['gen']) >= 1:
            print("Current generation is: ", args['gen'])
            x_train_all = [a+b for a,b in zip(x_train_all, x_test_kg_all)]
        print(x_test_all[0][0], x_test_all[1][0], x_test_all[2][0])

        # prepare for model
        loader, gt_label = dh.data_helper_bert(x_train_all, x_val_all, x_test_all, x_test_kg_all, x_test_kg_all, model_select, config)
        trainloader, valloader, testloader, trainloader2, kg_testloader1, kg_testloader2 = loader[0], loader[1], loader[2], loader[3], loader[4], loader[5]
        y_train, y_val, y_test, y_train2, y_test1, y_test2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3], gt_label[4], gt_label[5]
        y_val, y_test, y_train2, y_test1, y_test2 = y_val.to(device), y_test.to(device), y_train2.to(device), y_test1.to(device), y_test2.to(device)
        
        # train setup
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, int(args['gen']), float(args['dropout']),float(args['dropoutrest']))
        ####################################################################################
        #       model要load checkpoint
        ####################################################################################
        checkpoint_path = './checkpoint.pt'
        checkp = torch.load(checkpoint_path)
        model.load_state_dict(checkp)  
        print(100*"#")
        print("model loaded from checkpoint: {}".format(checkpoint_path))
        print(100*"#")        
        ####################################################################################        


        loss_function = nn.CrossEntropyLoss()
        sum_loss = []
        val_f1_average, val_f1_against, val_f1_favor = [], [], []
        test_f1_average, test_f1_against, test_f1_favor, test_kg = [], [], [], []

        # early stopping


        es_intermediate_step = len(trainloader)//args['savestep']
        patience = args['earlystopping_step']   # the number of iterations that loss does not further decrease
        # patience = es_intermediate_step   # the number of iterations that loss does not further decrease        
        early_stopping = EarlyStopping(patience, verbose=True)
        print(100*"#")
        # print("len(trainloader):",len(trainloader))
        # print("args['savestep']:",args['savestep'])
        print("early stopping occurs when the loss does not decrease after {} steps.".format(patience))
        print(100*"#")
        # print(bk)
        # init best val/test results
        best_train_f1macro = 0
        best_train_result = []
        best_val_f1macro = 0
        best_val_result = []
        best_test_f1macro = 0
        best_test_result = []

        best_val_loss = 100000
        best_val_loss_result = []
        best_test_loss = 100000
        best_test_loss_result = []
        # start training
        print(100*"#")
        print("clipgradient:",args['clipgradient']=='True')
        print(100*"#")


        model.eval()
        with torch.no_grad():
            preds_test, loss_test = model_utils.model_preds(valloader, model, device, loss_function,model_select)

        step = 0
        f1macro_test, result_one_sample_test = compute_performance(preds_test,y_val,'test',step, args, seed)

        best_test_loss_result = result_one_sample_test
        best_test_loss_result[0][0]='best test noun phrase'
        results_df = pd.DataFrame(best_test_loss_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_test_noun_phrase_df.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_test_noun_phrase_df.csv save, done!')


        with torch.no_grad():
            preds_test, loss_test = model_utils.model_preds(testloader, model, device, loss_function,model_select)
        f1macro_test, result_one_sample_test = compute_performance(preds_test,y_test,'test',step, args, seed)

        best_test_loss_result = result_one_sample_test
        best_test_loss_result[0][0]='best test claim'
        results_df = pd.DataFrame(best_test_loss_result)    
        print("results_df are:",results_df.head())
        results_df.to_csv('./best_loss_results_test_claim_df.csv',index=False, mode='a', header=False)    
        print('./best_loss_results_test_claim_df.csv save, done!')
        

        


if __name__ == "__main__":
    run_classifier()
