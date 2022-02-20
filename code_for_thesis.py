# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:13:56 2021

@author: AmayaGS
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:34:20 2021

@author: AmayaGS
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms

from matplotlib import pyplot as plt
from matplotlib import ticker as tc
import matplotlib.cm as cm

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report

import time
import os, os.path
import glob
import copy
import cv2

from collections import defaultdict

from collections import Counter

from torchsampler import ImbalancedDatasetSampler

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

plt.ion()  

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

import gc 
gc.enable()

# %% HELPER FUNCTIONS


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(18,2))
    plt.axis('off')
    plt.imshow(inp, interpolation='nearest')
    if title is not None:
        plt.title(title, fontsize=15)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

# Get a batch of training data
#inputs, classes = next(iter(dataloaders[TRAIN]))
#show_databatch(inputs, classes)

def visualize_model(vgg, num_images=6):
    was_training = vgg.training
    
    # Set model for evaluation
    vgg.train(False)
    vgg.eval() 
    
    images_so_far = 0

    for i, data in enumerate(dataloaders[TEST]):
        inputs, labels = data
        size = inputs.size()[0]
        
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        else:
            inputs, labels = inputs, labels
        
        outputs = vgg(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu())
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    vgg.train(mode=was_training) # Revert model back to original training state
    

def eval_model(vgg, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0
    
    #test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)
    
    f = open("test_y_probs_" + test_dir + '_' + test_split + '_' + cell_name + ".csv", "w")
    #f = open("test_y_probs_" + test_dir + '_' + test_split + '_' + ".csv", "w")
    f.write("QMULID,CD, Image,Prediction,Label,Prob0,Prob1\n")
    
    for i, data in enumerate(dataloaders[TEST]):
        #if i % 100 == 0:
            #print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data
        
        
        with torch.no_grad():
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs, labels

        outputs = vgg(inputs)
        
        probs = F.softmax(outputs, dim=1)
        #np_probs = [t.detach().numpy() for t in probs]
        np_probs = probs.detach().to('cpu').numpy()
        #print(np_probs[0][0])
        _, preds = torch.max(outputs.data, 1)
        #print(preds.item())
        sample_fname, _ = dataloaders[TEST].dataset.samples[i]
        qmul_id = str(sample_fname.split('\\')[12])
        sample_name = str(sample_fname.split('\\')[13])
        cd_name = cd.split('\\')[9]
        loss = criterion(outputs, labels)
        
        f.write("{}, {}, {}, {}, {}, {}, {}\n".format(qmul_id, cd_name, sample_name, preds.item(), labels.item(), np_probs[0][0], np_probs[0][1]))
        #f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(qmul_id, cd_name, sample_name, preds.item(), labels.item(), np_probs[0][0], np_probs[0][1], np_probs[0][2]))
        #f.write("{}, {}, {}, {}, {}, {}, {}\n".format(qmul_id, sample_name, preds.item(), labels.item(), np_probs[0][0], np_probs[0][1], np_probs[0][2]))
        
        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)
        
        
        
        del inputs, labels, outputs, preds, probs
        torch.cuda.empty_cache()
        
    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)
    
    f.close()
    

def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    history = defaultdict(list)
    
    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[TEST])
    
    vgg.train()
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)
        
        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                
            # Use half training dataset
            if i >= train_batches / 2:
                break
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            else:
                inputs, labels = inputs, labels
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
            
        #torch.save(vgg16.state_dict(), 'vgg16_fe_PEAC_HE' + test_dir + '_' + test_split + '.pt')
        
        
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train * 2 / dataset_sizes[TRAIN]
        
        vgg.train(False)
        
        vgg.eval()
            
        for i, data in enumerate(dataloaders[TEST]):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                
            inputs, labels = data
            
            with torch.no_grad():
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                else:
                    inputs, labels = inputs, labels
            
            optimizer.zero_grad()
            
            outputs = vgg(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data)
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / dataset_sizes[TEST]
        avg_acc_val = acc_val / dataset_sizes[TEST]
        
        history['train_acc'].append(avg_acc)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(avg_acc_val)
        history['val_loss'].append(avg_loss_val)
        
        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    vgg.load_state_dict(best_model_wts)
    
    return vgg, history

def plot_training_history(history):
    
  figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
  
  ax1.plot(history['train_acc'], label='train accuracy')
  ax1.plot(history['val_acc'], label='test accuracy')
  
  ax1.xaxis.set_major_locator(tc.MaxNLocator(integer=True))
  ax1.set_ylim([-0.05, 1.05])
  ax1.legend()
  ax1.set_ylabel('Accuracy')
  ax1.set_xlabel('Epoch')
  
  ax2.plot(history['train_loss'], label='train loss')
  ax2.plot(history['val_loss'], label='test loss')
  
  ax2.xaxis.set_major_locator(tc.MaxNLocator(integer=True))
  ax2.set_ylim([-0.05, 1.05])
  ax2.legend()
  ax2.set_ylabel('Loss')
  ax2.set_xlabel('Epoch')
  figure.suptitle('Training history ' + test_dir + ' ' + test_split + ' ' + cell_name )
  
  figure.savefig('Training history ' + test_dir + ' ' + test_split + ' ' + cell_name  + '.png', dpi=300)

def conf_matrix(model, TEST):
    
    nb_classes = len(image_datasets[TEST].classes)
    #im_root = dataloaders[TEST].dataset.root
    #im_version = str(im_root.split('\\')[6])
    #im_name = str(im_root.split('\\')[7])
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    
    model = model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders[TEST]):
            
                if use_gpu:
                    inputs, classes = inputs.cuda(), classes.cuda()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    predictions.extend(preds)
                    real_values.extend(classes)
                    
                    for t, p in zip(classes.view(-1), preds.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                            
    conf_matrix_np = confusion_matrix.numpy()
    conf_matrix = conf_matrix_np / conf_matrix_np.astype(np.float).sum(axis=1, keepdims=True)
    #print(conf_matrix)
                            
    # classification report output          
    predictions = torch.as_tensor(predictions).cpu()
    real_values = torch.as_tensor(real_values).cpu()
    
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    fig = plt.figure()
    colors = sns.color_palette("Blues", as_cmap=True)
    
    class_names = image_datasets[TEST].classes
    mat_shape = len(class_names)
    conf_matrix_per = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    count_matrix = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    #print(conf_matrix_per)
    
    #prepare annotations for the confusion matrix
    acc_class_per = (confusion_matrix/confusion_matrix.sum(1)[:, np.newaxis]).flatten()
    acc_class_per = ['{0:.2%}'.format(value) for value in
                          acc_class_per]
    cf_matrix_np = count_matrix.to_numpy()
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix_np.flatten()]
    
    labels = [f'{v1}\n{v2}' for v1, v2 in
              zip(acc_class_per, group_counts)]
    labels = np.asarray(labels).reshape(mat_shape, mat_shape)
    
    heatmap = sns.heatmap(conf_matrix_per, annot= labels, fmt='', cmap= colors)
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), ha='left', rotation=90, fontsize=9, )
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=11)
    heatmap.xaxis.set_tick_params(pad=15)
    heatmap.yaxis.set_tick_params(pad=15)
    
    plt.ylabel('true label', fontsize=15, loc='center', labelpad=20)
    plt.xlabel('predicted label', fontsize=15, loc='center', labelpad=20)
    plt.title('Confusion matrix ' + test_dir + ' ' + test_split + ' ' + cell_name , fontsize=12, loc='center')
    
    fig.savefig('conf_matrix' + test_dir + ' ' + test_split + ' ' + cell_name  +  '.png')
    
    return predictions, real_values

# %%

TRAIN = 'Train'
TEST = 'Test'

data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Resize((224, 224)),                            
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.01),
        transforms.ColorJitter(contrast=0.01), 
        transforms.ColorJitter(saturation=0.01),
        transforms.ColorJitter(hue=0.01)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      
    ]),
    TEST: transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# %%

path = r'C:\Users\AmayaGS\Documents\PhD\MRes\PEAC_IHC'
test_subdirs = ['test_50', 'test_40', 'test_33', 'test_25', 'test_20']
cd_subdirs = ['CD3', 'CD20' ,'CD68', 'CD138']
test_train = ['Test', 'Train']
folders_in_test_subdirs = {'test_50':2, 'test_40':2, 'test_33':3, 'test_25':4, 'test_20':5}

TRAIN = 'Train'
TEST = 'Test'

# %%
test_subdirs = [f.path for f in os.scandir(path) if f.is_dir()]

for subdir in test_subdirs:
    
    splits = [f.path for f in os.scandir(path) if f.is_dir()]
    
    for t in splits:
        
        split = [f.path for f in os.scandir(t) if f.is_dir()]
        
        for s in split:
            
            cds = [f.path for f in os.scandir(s) if f.is_dir()]
            
            for cd in cds:
                
                gc.collect()
                torch.cuda.empty_cache()
                
                data_dir = cd
        
                test_dir = data_dir.split('\\')[7]
                test_split = data_dir.split('\\')[8]
                cell_name = data_dir.split('\\')[9]
    
                image_datasets = {
                    x: datasets.ImageFolder(
                        os.path.join(data_dir, x), 
                        transform=data_transforms[x]
                    )
                    for x in [TRAIN, TEST]
                }
                
                dataloaders = {
                    TRAIN: torch.utils.data.DataLoader(
                        image_datasets[TRAIN], batch_size=1,
                        sampler = ImbalancedDatasetSampler(image_datasets[TRAIN])
                    ),
                    TEST: torch.utils.data.DataLoader(
                        image_datasets[TEST], batch_size=1, shuffle= False
                    )
                }
                
                print(test_dir, test_split, cell_name)
                #print(test_dir, test_split)
                # checking dataset properties
                #im_root = dataloaders[TRAIN].dataset.root
                #im_version = str(im_root.split('\\')[6])
                #im_cell = str(im_root.split('\\')[9])
                
                dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, TEST]}
                
                for x in [TRAIN, TEST]:
                    print("Loaded {} images under {}".format(dataset_sizes[x], x))
                    label_dict = dict(Counter(image_datasets[x].targets))
                    print(label_dict)
                    print(image_datasets[x].class_to_idx)
                    
                len_classes = dict(Counter(image_datasets[TRAIN].targets))
                
                class_size = []
                for cat in len_classes:
                    size = len_classes[cat]/ dataset_sizes[TRAIN]
                    class_size.append(size)
                    
                print(class_size)
                
                #print("Classes: ")
                class_names = image_datasets[TRAIN].classes
                #print(image_datasets[TRAIN].classes)
                
                labels = [label for _, label in dataloaders[TRAIN].dataset.imgs]
                figs, ax = plt.subplots()
                class_labels, counts = np.unique(labels, return_counts=True)
                
                colors = ['xkcd:dull red','xkcd:bluish green','xkcd:faded blue']
                
                ax.bar(class_labels, counts, color=colors)
                ax.tick_params(axis='y', colors='black', labelsize=12)
                ax.set_xticks(class_labels)
                ax.set_xticklabels(class_names, fontsize=12)
                
                plt.title('Distribution of classes ' + test_dir + ' ' + test_split + ' ' + cell_name, fontsize=13)
                #plt.title('Distribution of classes ' + test_dir + ' ' + test_split, fontsize=13)
                plt.ylabel('# Images', fontsize=13)
                plt.show()
                
                figs.savefig('Distribution of classes_' + test_dir + ' ' + test_split + ' ' + cell_name + '.png')
                #figs.savefig('Distribution of classes_' + test_dir + ' ' + test_split + '.png')
    
                #vgg16 = ToyModel(in_channels, out_channels, input_dim)
                vgg16 = models.vgg16_bn(pretrained=True)
                
                # Freeze training for all layers
                for param in vgg16.parameters():
                    param.require_grad = False
                
                # Newly created modules have require_grad=True by default
                num_features = vgg16.classifier[6].in_features
                features = list(vgg16.classifier.children())[:-1] # Remove last layer
                features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 3 outputs
                vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
                # #print(vgg16)
                
                if use_gpu:
                    vgg16.cuda() #.cuda() will move everything to the GPU side
                    
                criterion = nn.CrossEntropyLoss()
                
                optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.0001, momentum=0.9, weight_decay=0)
                #optimizer_ft = optim.Adam(vgg16.parameters())
                exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
                
                vgg16, history = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)
                torch.save(vgg16.state_dict(), 'vgg16_fe_PEAC_2class' + test_dir + '_' + test_split + '_' + cell_name + '.pt')
                #torch.save(vgg16.state_dict(), 'vgg16_fe_PEAC_2class' + test_dir + '_' + test_split + '.pt')
                
                plot_training_history(history)
                
                y_pred, y_test = conf_matrix(vgg16, TEST)
                clsf_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=class_names, output_dict=True)).transpose()
                clsf_report.to_csv('classification_report_' + test_dir + '_' + test_split + '_' + cell_name + '.csv', index=True)
                #clsf_report.to_csv('classification_report_2class' + test_dir + '_' + test_split + '.csv', index=True)
                print(clsf_report)
                eval_model(vgg16, criterion)
    

# %%

# VOTING SYSTEMS

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import re

# %%

path = r'C:\Users\AmayaGS\Documents\PhD\MRes\results_33_2_class\y_test_probs_lympho_pauci'
files = os.listdir(path)

# %%

#df = pd.read_csv(r"C:\Users\AmayaGS\Documents\PhD\MRes\test_y_cd68_probs.csv")
#df = pd.read_csv(r"C:\Users\AmayaGS\Documents\PhD\MRes\test_y_cd20_prob.csv")
#df = pd.read_csv(r"C:\Users\AmayaGS\Documents\PhD\MRes\test_y_cd138_probs.csv")
#df = pd.read_csv(r"C:\Users\AmayaGS\Documents\PhD\MRes\test_y_cd3_probs.csv")

for file in files:
    
    celltype = re.split(r'_', file)[6][:-4]
    test_split = re.split(r'_', file)[5]
    
    df = pd.read_csv(file)
    
    # adding 3 columns to dataframe for different voting metrics
    df['vote'] = 0 # majority voting
    df['max_prob'] = 0 # soft voting 
    df['max_prob_vote'] = 0 # soft voting + majority voting for ex aequo results
    
    # checking we have the right subjects
    subjects = set(df.QMULID)
    
    #  majority voting. 
    # returns the mode of the different classes predicted for each image of a given patient. 
    for i in range(len(df)):
        
       df.loc[i,'vote'] = df[df['QMULID'] == df['QMULID'][i]]['Prediction'].mode()[0]  #choose first in list in case of ex-aequo results
    
    # passing the results of majority voting to the soft voting + majority voting decision for ex aequo results. 
    df['max_prob_vote'] = df['vote'] 

    # soft voting. 
    for id in subjects:
        
        prob0 = df.loc[df['QMULID']==id, 'Prob0'].sum()  # sum of probabilities per ID for each class. 
        prob1 = df.loc[df['QMULID']==id, 'Prob1'].sum() 
        prob2 = df.loc[df['QMULID']==id, 'Prob2'].sum()
        
        maxProbClass = np.array([prob0,prob1,prob2]).argmax() # returns the max prob sum. 
        
        df.loc[df['QMULID']==id,'max_prob'] = maxProbClass # passing result to max prob column.

    # combined voting
    for id in subjects:
        
        if len(df.loc[df['QMULID']==id,'Prediction'].mode()) > 1:
            df.loc[df['QMULID']==id,'max_prob_vote'] = df.loc[df['QMULID']==id,'max_prob'].iloc[0]    
    
    # which method performs better
    scoreProb = 0
    scoreVote = 0
    scoreVoteProb = 0 
    size = len(subjects)    
    
    for id in subjects:
    
        if df.loc[df['QMULID']==id,'Label'].iloc[0] == df.loc[df['QMULID']==id,'max_prob'].iloc[0]:
            scoreProb += 1
        if df.loc[df['QMULID']==id,'Label'].iloc[0] == df.loc[df['QMULID']==id,'vote'].iloc[0]:
            scoreVote += 1  
        if df.loc[df['QMULID']==id,'Label'].iloc[0] == df.loc[df['QMULID']==id,'max_prob_vote'].iloc[0]:
            scoreVoteProb += 1
    
    print(file.split('_')[5], file.split('_')[6])
    print(scoreProb/size, scoreVote/size, scoreVoteProb/size)

    
    df_vote_score = df[['QMULID', 'vote', 'Label', 'max_prob']]
    df_patient_level = df_vote_score.groupby('QMULID').agg('min')
    df_patient_level.to_csv('Majority_vote_prediction_' + test_split + '_' + celltype + '.csv', index=True)
    
    
    clsf_report = pd.DataFrame(classification_report(np.array(df_patient_level['Label']), np.array(df_patient_level['vote']), output_dict=True)).transpose()
    clsf_report.to_csv('classification_report_patient_level_' + test_split + '_' +  celltype + '.csv', index=True)
    print(clsf_report)
    print(confusion_matrix(np.array(df_patient_level['Label']), np.array(df_patient_level['vote'])))

# %%

# voting at the CD level
# Test 1
import itertools

split_test = ['Test 1', 'Test 2', 'Test 3']
celltype = ['CD3', 'CD20' ,'CD68', 'CD138']
#set_sizes = [4,3,2]
set_sizes = [1]
base_set = {'CD3', 'CD20' ,'CD68', 'CD138'}

for set_size in set_sizes:
    cell_sets = itertools.combinations(base_set, set_size)
    
    for cell_set in cell_sets:
        
        for split in split_test:
            
            df_list = []
            
            for cell in cell_set:
        
                df_list.append(pd.read_csv(r"C:/Users/AmayaGS/Documents/PhD/MRes/Results_3class_cross_val/test_33_results/Majority_vote_prediction_" + split + '_' + cell + ".csv"))
           
            df_all = pd.concat(df_list, ignore_index=True)


            df = df_all 
            df['majority_vote'] = 0 #
            
            subjects = set(df.QMULID)
  
            
            #  majority voting. 
            # returns the mode of the different classes predicted for each image of a given patient. 
            for i in range(len(df)):
                
               df.loc[i,'majority_vote'] = df[df['QMULID'] == df['QMULID'][i]]['vote'].mode()[0]  
      
               
            scoreVote = 0
            
            size = len(subjects)    
            
            for id in subjects:
            
                if df.loc[df['QMULID']==id,'Label'].iloc[0] == df.loc[df['QMULID']==id,'vote'].iloc[0]:
                    scoreVote += 1  
                    
            print(scoreVote/size)
      
            
            df_vote_score = df[['QMULID', 'majority_vote', 'Label']]
            df_patient_level = df_vote_score.groupby('QMULID').agg('min')
            df_patient_level.to_csv('Majority_vote_prediction_all+ '.csv', index=True)
            
        
            clsf_report = pd.DataFrame(classification_report(np.array(df_patient_level['Label']), np.array(df_patient_level['majority_vote']), output_dict=True)).transpose()
            clsf_report.to_csv('classification_report_patient_level' + split + '_'.join(cell_set) + '.csv', index=True)
            print(clsf_report)
            conf_matrix_all = confusion_matrix(np.array(df_patient_level['Label']), np.array(df_patient_level['majority_vote']))
            conf_matrix_norm = conf_matrix_all/np.sum(conf_matrix_all, axis=1)
            print(conf_matrix_norm)
            
                
            
            acc_class_per = conf_matrix_norm.flatten()
            acc_class_per = ['{0:.2%}'.format(value) for value in
                                  acc_class_per]
            group_counts = ['{0:0.0f}'.format(value) for value in
                            conf_matrix_all.flatten()]
            
            labels = [f'{v1}\n{v2}' for v1, v2 in
                      zip(acc_class_per, group_counts)]
            
            mat_shape = len(class_names)
            labels = np.asarray(labels).reshape(mat_shape, mat_shape)
            
         
            
            pd_conf_matrix_norm = pd.DataFrame(conf_matrix_norm, index=class_names, columns=class_names)
            sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
            sns.set_style("white")
            sns.set_context("paper", font_scale=1.0)
            fig = plt.figure()
            heatmap = sns.heatmap(pd_conf_matrix_norm, annot= labels, fmt='', cmap= 'Blues')
            
            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), ha='left', rotation=90, fontsize=9, )
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=10)
            heatmap.xaxis.set_tick_params(pad=15)
            heatmap.yaxis.set_tick_params(pad=15)
            
            plt.ylabel('true label', fontsize=15, loc='center', labelpad=20)
            plt.xlabel('predicted label', fontsize=15, loc='center', labelpad=20)
            plt.title('Majority voting ' + split + ' ' + '_'.join(cell_set), fontsize=15, loc='center')
            
            fig.savefig('conf_matrix_all_cds' + split + '_'.join(cell_set) + '.png')

# %%


cd68 = pd.read_csv('Majority_vote_prediction_Test 1_CD68.csv')
cd20 = pd.read_csv('Majority_vote_prediction_Test 1_CD20.csv')
cd138 = pd.read_csv('Majority_vote_prediction_Test 1_CD138.csv')

# %%

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)
fig = plt.figure()
colors = sns.color_palette("Blues", as_cmap=True)

   
clsf_report = pd.DataFrame(classification_report(np.array(cd68['vote']), np.array(cd138['vote']), output_dict=True)).transpose()
print(clsf_report)
print(confusion_matrix(np.array(cd20['vote']), np.array(cd138['vote'])))

pd_conf_matrix_norm = pd.DataFrame(confusion_matrix(np.array(cd68['vote']), np.array(cd138['vote'])), index=class_names, columns=class_names)
heatmap = sns.heatmap(pd_conf_matrix_norm, annot= True, fmt='', cmap= 'Blues')

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), ha='left', rotation=90, fontsize=9)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=11)
plt.ylabel('CD68 prediction', fontsize=10, loc='center', labelpad=20)
plt.xlabel('CD138 prediction', fontsize=10, loc='center', labelpad=20)
plt.title('Class predictions for CD68 vs CD138', fontsize=15, loc='center')

fig.savefig('Class prediction for CD68 and CD138.png')


# %%

peac_imid_data = pd.read_csv("PEAC_IMID_data.csv")
Pathotype = peac_imid_data[['QMULID', 'Pathotype']]
#Pathotype.set_index('QMULID', inplace=True)
Pathotype = Pathotype.dropna()

# %%

import os
import shutil
import random
from distutils.dir_util import copy_tree, create_tree
import numpy as np
import pandas as pd
import random


# %%

# DIR COPY

import os
import shutil
import random
from distutils.dir_util import copy_tree, create_tree
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


# %%

# ### Import PEAC clinical data

peac_imid_data = pd.read_csv("PEAC_IMID_data.csv")

# %%

Pathotype = peac_imid_data[['QMULID', 'Pathotype']]
Pathotype.set_index('QMULID', inplace=True)
Pathotype = Pathotype.dropna()

# %%

# %%

pathotype_labels, pathotype_counts = np.unique(Pathotype, return_counts=True)
norm_counts = pathotype_counts/np.sum(pathotype_counts, axis=0)

fig, ax = plt.subplots()
#colors = ['tab:red', 'mediumseagreen', 'tab:carolina blue']
#colors = ['#5cb85c','#5bc0de','#d9534f']
colors = ['xkcd:dull red','xkcd:bluish green','xkcd:faded blue']

ax.bar(pathotype_labels, pathotype_counts, color= colors)
ax.tick_params(axis='y', colors='black', labelsize=12)
ax.set_xticks(pathotype_labels)
ax.set_xticklabels(pathotype_labels, fontsize=12)

for p, per in zip(ax.patches, norm_counts):
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f"{per:.1%}", (x + width/2, y + height*0.8), ha='center', fontsize=12)

plt.title('Distribution of patients by class', fontsize=13)
plt.ylabel('# Patients', fontsize=13, labelpad=15)
plt.show()

#fig.savefig('Distribution of patients.png')

# %%

class_weights = {"Diffuse-Myeloid": norm_counts[0], "Lympho-Myeloid": norm_counts[1], "Pauci-Fibroid": norm_counts[2]}

total = len(Pathotype)
test_50 = int(total * 0.5)
test_40 = int(total * .40)
test_33 = int(total * .333)
test_25 = int(total * .25)
test_20 = int(total * .20)

splits = [test_50, test_40, test_33, test_25, test_20]
name_splits = ['test_50', 'test_40', 'test_33', 'test_25', 'test_20']

for split, name in zip(splits, name_splits):
    
    print(split)
    if name == 'test_50':
        
        df = pd.DataFrame(Pathotype)
        
        df1 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df1.index)
        df2 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
    
        df1.to_csv(name + '_df1.csv')
        df2.to_csv(name + '_df2.csv')
        
    if name == 'test_40':
        
        df = pd.DataFrame(Pathotype)
        
        df1 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df1.index)
        df2 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
    
        df1.to_csv(name + '_df1.csv')
        df2.to_csv(name + '_df2.csv')
    
    if name == 'test_33':
        
        df = pd.DataFrame(Pathotype)

        df1 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df1.index)
        df2 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df2.index)
        df3 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        
        df1.to_csv(name + '_df1.csv')
        df2.to_csv(name + '_df2.csv')
        df3.to_csv(name + '_df3.csv')
        
    if name == 'test_25':
         
        df = pd.DataFrame(Pathotype)
         
        df1 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df1.index)
        df2 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df2.index)
        df3 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df3.index)
        df4 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
     
        df1.to_csv(name + '_df1.csv')
        df2.to_csv(name + '_df2.csv')
        df3.to_csv(name + '_df3.csv')
        df4.to_csv(name + '_df4.csv')
        
    if name == 'test_20':
         
        df = pd.DataFrame(Pathotype)
         
        df1 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df1.index)
        df2 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df2.index)
        df3 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df3.index)
        df4 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])
        df = df.drop(df4.index)
        df5 = pd.concat([df[df['Pathotype'] == k].sample(int(v * split), replace=False) for k, v in class_weights.items()])        
         
        df1.to_csv(name + '_df1.csv')
        df2.to_csv(name + '_df2.csv')
        df3.to_csv(name + '_df3.csv')
        df4.to_csv(name + '_df4.csv')
        df5.to_csv(name + '_df5.csv')


# ### Import PEAC clinical data

#peac_imid_data = pd.read_csv("QMUL_SAMID_database.csv")
peac_imid_data = pd.read_csv("PEAC_IMID_data.csv")

# %%

Pathotype = peac_imid_data[['QMULID', 'Pathotype']]
Pathotype.set_index('QMULID', inplace=True)
Pathotype = Pathotype.dropna()
id_dict = Pathotype.to_dict()['Pathotype']

# %%

qmul_ids = pd.DataFrame(Pathotype.index)

# %%

path = r'C:\Users\AmayaGS\Documents\PhD\MRes\subsetted_HE'
list_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
 
test_subdirs = ['test_33']
cd_subdirs = ['CD3', 'CD20' ,'CD68', 'CD138']
test_train = ['Test', 'Train']
classes = set(np.unique(Pathotype))
subfolder_am = {'test_33':3}
root_path = r'C:\Users\AmayaGS\Documents\PhD\MRes\PEAC_HE'

#%%

for test_subdir in test_subdirs:
    os.mkdir(root_path + '\\' + test_subdir)
    
    for i in range(1, subfolder_am[test_subdir] + 1):
        
        local_dir = root_path + '\\' + test_subdir + '\\' + 'Test ' + str(i)
        os.mkdir(local_dir)
        
        for cd_subdir in cd_subdirs:
            os.mkdir(local_dir + '\\' + cd_subdir )
            
            for split in test_train:
                os.mkdir(local_dir + '\\' + cd_subdir + '\\' + split )
                    
                for clazz in classes:
                    copy_dir = local_dir + '\\' + cd_subdir + '\\' + split + '\\' + clazz 
                    os.mkdir(copy_dir)
                
                #test_subdirs_to_copy = [d + '_IHC' for d in list(pd.read_csv(test_subdir + '_df' + str(i) +'.csv')['QMULID'])]
                test_subdirs_to_copy = [d for d in list(pd.read_csv(test_subdir + '_df' + str(i) +'.csv')['QMULID'])]
                if split == 'Test':
                    
                    for subdir in test_subdirs_to_copy:
                        source = path + '\\' + subdir
                        target = local_dir + '\\' + cd_subdir + '\\' + split + '\\' + id_dict[subdir] + '\\' + subdir
                        os.mkdir(target)
                        allfiles = os.listdir(source)
                        #filtered_files = [f for f in allfiles if (cd_subdir in f or cd_subdir.lower() in f) ]
                        #for file in filtered_files:
                        for file in allfiles:
                            shutil.copyfile(source + '\\' + file,  target + '\\' + file )
                else:
                    #for subdir in [k + '_IHC' for k in id_dict.keys()]:
                    for subdir in [k for k in id_dict.keys()]:
                        if subdir not in test_subdirs_to_copy:
                            source = path + '\\' + subdir
                            target = local_dir + '\\' + cd_subdir + '\\' + split + '\\' + id_dict[subdir] + '\\' + subdir
                            os.mkdir(target)
                            allfiles = os.listdir(source)
                            #filtered_files = [f for f in allfiles if (cd_subdir in f or cd_subdir.lower() in f) ]
                            #for file in filtered_files:
                            for file in allfiles:
                                shutil.copyfile(source + '\\' + file, target + '\\' + file)                    