import os
import gc
import sys
import json
import random
import subprocess
import numpy as np

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

import _pickle as cPickle
import logging
import argparse


parser = argparse.ArgumentParser(description='Testing the regressors')

parser.add_argument('--config_path', type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir', type=str,
                    help=' The directory to the output folder')

args = parser.parse_args()

out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

logging.basicConfig(filename=os.path.join(args.out_dir, "test_log.txt"),
                    level=logging.INFO, filemode="w")

# Loads a json configuration file (test_config.json)
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)

# Saves a json configuration file (test_config.json) in the experiment folder
with open(os.path.join(args.out_dir,'test_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

random.seed(config_dict["random_seed"])
np.random.seed(config_dict["random_seed"])

if config_dict["gpu_num"] != -1:
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config_dict["gpu_num"])
    args.use_cuda = True
else:
    args.use_cuda = False

import torch

args.use_cuda = args.use_cuda and torch.cuda.is_available()

torch.manual_seed(config_dict["seed"])
if args.use_cuda:
    torch.cuda.manual_seed(config_dict["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Testing with CUDA')

from scorer import *
from classes import *
from eval_utils import *
from model_utils import *

def test_model(test_set):
    '''
    Loads trained event and entity models and test them on the test set
    :param test_set: a Corpus object, represents the test split
    '''
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    cd_event_model = load_check_point(config_dict["cd_event_model_path"])
    cd_entity_model = load_check_point(config_dict["cd_entity_model_path"])

    cd_event_model.to(device)
    cd_entity_model.to(device)

    doc_to_entity_mentions = load_entity_wd_clusters(config_dict)

    scores_file = open(os.path.join(args.out_dir, 'Scores.txt'), 'w')



    config_dict["entity_merge_threshold"] = 0.5
    config_dict["event_merge_threshold"] = 0.5
    tag = "55"
    event_f1, entity_f1 = test_models(tag,False,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))

    config_dict["entity_merge_threshold"] = 0.6
    config_dict["event_merge_threshold"] = 0.5
    tag = "65"
    event_f1, entity_f1 = test_models(tag,False,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))

    config_dict["entity_merge_threshold"] = 0.5
    config_dict["event_merge_threshold"] = 0.6
    tag = "56"
    event_f1, entity_f1 = test_models(tag,False,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))

    config_dict["entity_merge_threshold"] = 0.6
    config_dict["event_merge_threshold"] = 0.6
    tag = "66"
    event_f1, entity_f1 = test_models(tag,False,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))
                      
    config_dict["entity_merge_threshold"] = [0.9,0.8,0.7,0.6,0.5]
    config_dict["event_merge_threshold"] = [0.9,0.8,0.7,0.6,0.5]
    tag = "s5s5"
    event_f1, entity_f1 = test_models(tag,True,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))
                      
    config_dict["entity_merge_threshold"] = [0.9,0.8,0.7,0.6,0.6]
    config_dict["event_merge_threshold"] = [0.9,0.8,0.7,0.6,0.5]
    tag = "s6s5"
    event_f1, entity_f1 = test_models(tag,True,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))
                      
    config_dict["entity_merge_threshold"] = [0.9,0.8,0.7,0.6,0.5]
    config_dict["event_merge_threshold"] = [0.9,0.8,0.7,0.6,0.6]
    tag = "s5s6"
    event_f1, entity_f1 = test_models(tag,True,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))
                      
    config_dict["entity_merge_threshold"] = [0.9,0.8,0.7,0.6]
    config_dict["event_merge_threshold"] = [0.9,0.8,0.7,0.6]
    tag = "s6s6"
    event_f1, entity_f1 = test_models(tag,True,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))
                      
    config_dict["entity_merge_threshold"] = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
    config_dict["event_merge_threshold"] = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
    tag = "s5i5"
    event_f1, entity_f1 = test_models(tag,True,test_set, cd_event_model, cd_entity_model, device, config_dict, write_clusters=True, out_dir=args.out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions,analyze_scores=True)
    scores_file.write('Experiment : {} Event F1: {} Entity F1: {}\n'.format(tag, event_f1, entity_f1))
    
    
    scores_file.close()


def main():
    '''
    This script loads the trained event and entity models and test them on the test set
    '''
    print('Loading test data...')
    logging.info('Loading test data...')
    with open(config_dict["test_path"], 'rb') as f:
        test_data = cPickle.load(f)

    print('Test data have been loaded.')
    logging.info('Test data have been loaded.')

    test_model(test_data)


if __name__ == '__main__':

    main()