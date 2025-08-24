import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from thop import profile
from datetime import datetime


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type == 'CoT':
        from models.model_CoT import CoT, CONFIGS
        config = CONFIGS[args.model_type]
        config.select_ratio = args.select_ratio
        config.transformer.num_layers = args.layer_num
        config.transformer.num_heads = args.head_num
        model_dict = {'config': config, 'img_size': args.img_size, 'num_classes':args.n_classes, 'zero_head':True, 'smoothing_value':args.smoothing_value}
        model = CoT(**model_dict)
        if(args.pretrained_dir != None):
            model.load_from(np.load(args.pretrained_dir))

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
        # pass
    model.eval()
    return model

def eval(mode, dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, mode=args.mode)
    patient_results, test_error, auc, test_f1, df, acc_logger, attention_result, test_patch_error, test_patch_f1 = summary(mode, model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('f1: ', test_f1)
    print('patch_arror:', test_patch_error)
    print('patch_f1:', test_patch_f1)

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)

    return model, patient_results, test_error, auc, test_f1, df, each_class_acc, attention_result, test_patch_error, test_patch_f1

def summary(mode, model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    test_patch_error = 0.
    test_patch_f1 = 0.
    flops = 0.
    params = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    all_pred = []
    all_label = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    attention_result = {}

    if(mode == 'clam'):
        a = datetime.now()
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                logits, Y_prob, Y_hat, A, results_dict= model(data)
                each_flops, each_params = profile(model, inputs=(data, ))

            acc_logger.log(Y_hat, label)

            probs = Y_prob.cpu().numpy()

            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()

            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

            attention_result[slide_id] = A

            error = calculate_error(Y_hat, label)
            test_error += error
            test_f1 = f1_score(all_label, all_pred, average='macro')
            flops += each_flops
            params += each_params

            del data

        test_error /= len(loader)
        flops /= len(loader)
        params /= len(loader)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')
        b = datetime.now()
        print('Time = ' + str((b - a).seconds/len(loader)) + 's')

        aucs = []
        if len(np.unique(all_labels)) == 1:
            auc_score = -1

        else:
            if args.n_classes == 2:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                for class_idx in range(args.n_classes):
                    if class_idx in all_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                        aucs.append(auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                if args.micro_average:
                    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                    auc_score = auc(fpr, tpr)
                else:
                    auc_score = np.nanmean(np.array(aucs))

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)
        return patient_results, test_error, auc_score, test_f1, df, acc_logger, attention_result, 0, 0

    elif(mode == 'transformer'):
        a = datetime.now()
        text_prompt_feature = torch.load(args.text_prompt_feature_path)
        for batch_idx, (data_s, coord_s, data_l, coord_l, label, patch_label, each_slide_id) in enumerate(loader):
            data_s, coord_s, data_l, coord_l, label, patch_label, each_slide_id = data_s.to(device), coord_s.to(device), data_l.to(device), \
                                                                        coord_l.to(device), label.to(device), patch_label.to(device), each_slide_id.to(device)
            text_prompt_feature = text_prompt_feature.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                logits, Y_prob, Y_hat, Y_hat_patch, A, _ = model(data_s, coord_s, data_l, coord_l, patch_label, label, text_prompt_feature, each_slide_id)

            acc_logger.log(Y_hat, label)

            probs = Y_prob.cpu().numpy()

            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()

            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

            attention_result[slide_id] = A

            error = calculate_error(Y_hat, label)
            test_error += error

        test_f1 = f1_score(all_label, all_pred, average='macro')
        test_error /= len(loader)
        test_patch_error /= len(loader)
        test_patch_f1 /= len(loader)

        flops /= len(loader)
        params /= len(loader)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')
        b = datetime.now()
        print('Time = ' + str((b - a).seconds/len(loader)) + 's')


        aucs = []
        if len(np.unique(all_labels)) == 1:
            auc_score = -1

        else:
            if args.n_classes == 2:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                for class_idx in range(args.n_classes):
                    if class_idx in all_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                        aucs.append(auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                if args.micro_average:
                    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                    auc_score = auc(fpr, tpr)
                else:
                    auc_score = np.nanmean(np.array(aucs))

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)
        return patient_results, test_error, auc_score, test_f1, df, acc_logger, attention_result, test_patch_error, test_patch_f1



def plot_confusion_matrix(cm, classes, save_path,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()

