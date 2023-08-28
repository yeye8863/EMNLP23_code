import preprocessor as p 
import re
import wordninja
import csv
import pandas as pd
from utils import augment

# Data Loading
def load_data(filename,mode,trainvaltest):

    concat_text = pd.DataFrame()

    if mode=='train_en_test_en': 
        encoding='ISO-8859-1'
        label_words=['AGAINST','FAVOR','NONE']

    df = pd.read_csv(filename)
    print(filename,"df.columns:",df.columns)
    try:
        raw_text = pd.read_csv(filename,usecols=['Text'], encoding=encoding)
    except:
        raw_text = pd.read_csv(filename,usecols=['Tweet'], encoding=encoding)

    raw_target = pd.read_csv(filename,usecols=['Target 1'], encoding=encoding)
    raw_label = pd.read_csv(filename,usecols=['Stance 1'], encoding=encoding)
    seen = pd.read_csv(filename,usecols=['seen?'], encoding=encoding)
    if 'data_subtaskB' in filename:
        inuse = pd.read_csv(filename,usecols=['In Use'], encoding=encoding)
        label = pd.DataFrame.replace(raw_label,label_words, [0,1,2])
        concat_text = pd.concat([raw_text, label, raw_target, seen, inuse], axis=1)
    else:
        label = pd.DataFrame.replace(raw_label,label_words, [0,1,2])
        concat_text = pd.concat([raw_text, label, raw_target, seen], axis=1)        
    concat_text.rename(columns={'Stance 1':'Stance','Target 1':'Target'}, inplace=True)


    if 'data_subtaskB' in filename:
        print(200*"#")
        print("before filter in use==0:",len(concat_text))
        concat_text = concat_text[concat_text['In Use'] != 0]
        print("after filter in use==0:",len(concat_text))
        print(200*"#")

    if 'train' not in filename:
        concat_text = concat_text[concat_text['seen?'] != 1] 
    print(filename,"concat_text.columns:",concat_text.columns)
    return concat_text

# Data Cleaning
def data_clean(strings, norm_dict):
    
    p.set_options(p.OPT.URL,p.OPT.EMOJI,p.OPT.RESERVED)
    clean_data = p.clean(strings) # using lib to clean URL,hashtags...
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+",clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0]) # separate hashtags
    clean_data = [j for i in clean_data for j in i]

    return clean_data


# Clean All Data
def clean_all(filename, norm_dict, mode,trainvaltest,prompt_index):
    
    concat_text = load_data(filename, mode,trainvaltest) # load all data as DataFrame type    
    try:
        raw_data = concat_text['Text'].values.tolist() # convert DataFrame to list ['string','string',...]
    except:
        raw_data = concat_text['Tweet'].values.tolist() # convert DataFrame to list ['string','string',...]

    label = concat_text['Stance'].values.tolist()
    x_target = concat_text['Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]
    
    for i in range(len(raw_data)): 
        # if "raw_test_all_onecol" in filename: ### don't add prompt for test set
        if "fake_filename" in filename: ### don't add prompt for test set
            x_target[i] = x_target[i] 
        else: ### only add prompt for train/val set
            if prompt_index==0:
                x_target[i] = x_target[i]    
            if prompt_index==1:
                x_target[i] = 'I am in favor of '+x_target[i]+"!"
            elif prompt_index==2:
                x_target[i] = 'I support '+x_target[i]+"!"
            elif prompt_index==3:
                x_target[i] = 'I agree with '+x_target[i]+"!"
            elif prompt_index==4:
                x_target[i] = 'This entails '+x_target[i]+"!"
            elif prompt_index==5:
                x_target[i] = 'The above text entails '+x_target[i]+"!"
            elif prompt_index==6:
                x_target[i] = 'The premise entails '+x_target[i]+"!"
            elif prompt_index==7:
                x_target[i] = 'The premise entails '+x_target[i]+"."
            elif prompt_index==8:
                x_target[i] = 'The premise has entailment relation with '+x_target[i]+"!"
            elif prompt_index==9:
                x_target[i] = 'This implies the entailment relation with '+x_target[i]+"!"
            elif prompt_index==10:
                x_target[i] = 'The premise implies the entailment relation with '+x_target[i]+"!"
            elif prompt_index==11:
                x_target[i] = 'The premise has the entailment relation with the hypothesis '+x_target[i]+"!"
            elif prompt_index==12:
                x_target[i] = 'The premise entails the hypothesis '+x_target[i]+"!"



            elif prompt_index==101:
                # print("x_target[i] before:",x_target[i])
                # print("label[i] before:",label[i])
                x_target[i] = 'I am against with '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1
                # print("x_target[i] after:",x_target[i])
                # print("label[i] after:",label[i])
                # print(bk)
            elif prompt_index==102:
                x_target[i] = 'I oppose '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1           
            elif prompt_index==103:
                x_target[i] = 'I disagree with '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1           
            elif prompt_index==104:
                x_target[i] = 'This contradicts '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==105:
                x_target[i] = 'The above text contradicts '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==106:
                x_target[i] = 'The premise contradicts '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==107:
                x_target[i] = 'The premise contradicts '+x_target[i]+"."
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==108:
                x_target[i] = 'The premise has contradiction relation with '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==109:
                x_target[i] = 'This implies the contradiction relation with '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==110:
                x_target[i] = 'The premise implies the contradiction relation with '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==111:
                x_target[i] = 'The premise has the contradiction relation with the hypothesis '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  
            elif prompt_index==112:
                x_target[i] = 'The premise contradicts the hypothesis '+x_target[i]+"!"
                if label[i]==1:
                    label[i]=0
                elif label[i]==0:
                    label[i]=1  

        clean_data[i] = data_clean(raw_data[i],norm_dict) # clean each tweet text [['word1','word2'],[...],...]
        x_target[i] = data_clean(x_target[i],norm_dict)

    avg_ls = sum([len(x) for x in clean_data])/len(clean_data)


    print(100*"#")
    print("average length: ", avg_ls)
    print("num of subset: ", len(label))
    print("x_target[0]:",x_target[0])
    print(100*"#")
    return clean_data,label,x_target



# Clean All Data
def clean_all2(filename, norm_dict, mode,trainvaltest):
    
    concat_text = load_data(filename, mode,trainvaltest) # load all data as DataFrame type
    raw_data = concat_text['Text'].values.tolist() # convert DataFrame to list ['string','string',...]
    label = concat_text['Stance'].values.tolist()
    x_target = concat_text['Target'].values.tolist()
    clean_data = [None for _ in range(len(raw_data))]
    
    for i in range(len(raw_data)):
        # clean_data[i] = data_clean(raw_data[i],norm_dict) # clean each tweet text [['word1','word2'],[...],...]
        # x_target[i] = data_clean(x_target[i],norm_dict)
        clean_data[i] = raw_data[i] # clean each tweet text [['word1','word2'],[...],...]
        x_target[i] = x_target[i]
    avg_ls = sum([len(x) for x in clean_data])/len(clean_data)
    
    print("average length: ", avg_ls)
    print("num of subset: ", len(label))
    
    return clean_data,label,x_target