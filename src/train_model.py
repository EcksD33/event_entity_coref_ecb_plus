import os
import sys
import json
import shutil
import random
import logging
import argparse
import numpy as np
import pickle

# os.environ['GPU_DEBUG'] = "0"
# import gpu_profiling.gpu_profile as gpup
# sys.settrace(gpup.gpu_profile)


parser = argparse.ArgumentParser(description="Training a regressor")
parser.add_argument("--config_path", type=str,
                    help=" The path configuration json file")
parser.add_argument("--out_dir", type=str,
                    help=" The directory to the output folder")
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

logging.basicConfig(filename=os.path.join(args.out_dir, "train_log.txt"),
                    level=logging.DEBUG, filemode="a")

# Load json config file
with open(args.config_path, "r") as js_file:
    config_dict = json.load(js_file)

# Copy config file to out_dir for reproducibility
shutil.copyfile(args.config_path, os.path.join(args.out_dir, "train_config.json"))

# Copy experiment-dependent function implementation
try:
    with open(os.path.join(args.out_dir, "mention_pairs_to_input.py"), "x") as pyfile:
        from inspect import getsource
        from all_models.model_utils import mention_pair_to_model_input
        pyfile.write(getsource(mention_pair_to_model_input))
        print("'mention_pairs_to_input.py' has been saved.")
except FileExistsError:
    print("WARNING: 'mention_pairs_to_input.py' already exists in out folder.")

random.seed(config_dict["random_seed"])
np.random.seed(config_dict["random_seed"])

if config_dict["gpu_num"] != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["gpu_num"])
    args.use_cuda = True
else:
    args.use_cuda = False

import torch

args.use_cuda = args.use_cuda and torch.cuda.is_available()

import all_models.model_utils as MU
import all_models.model_factory as MF

torch.manual_seed(config_dict["seed"])
if args.use_cuda:
    torch.cuda.manual_seed(config_dict["seed"])
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    print("Training with CUDA")


def train_model(train_set, dev_set):
    '''
    Initializes models, optimizers and loss functions, then, it runs the training procedure that
    alternates between entity and event training and clustering on the train set.
    After each epoch, it runs the inference procedure on the dev set and calculates
    the B-cubed measure and use it to tune the model and its hyper-parameters.
    Saves the entity and event models that achieved the best B-cubed scores on the dev set.
    :param train_set: a Corpus object, representing the train set.
    :param dev_set: a Corpus object, representing the dev set.
    '''
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    doc_to_entity_mentions = MU.load_entity_wd_clusters(config_dict)  # loads predicted WD entity coref chains from external tool

    print("Setting up...")
    logging.info("Create new models...")
    MF.factory_load_embeddings(config_dict)  # loading pre-trained embeddings before creating new models
    cd_event_model = MF.create_model(config_dict)
    cd_entity_model = MF.create_model(config_dict)

    cd_event_optimizer = MF.create_optimizer(config_dict, cd_event_model)
    cd_entity_optimizer = MF.create_optimizer(config_dict, cd_entity_model)

    cd_event_loss = MF.create_loss(config_dict)
    cd_entity_loss = MF.create_loss(config_dict)

    topics = train_set.topics  # Use the gold sub-topics

    topics_num = len(topics.keys())
    event_best_dev_f1 = 0
    entity_best_dev_f1 = 0
    best_event_epoch = 0
    best_entity_epoch = 0

    patience_counter = 0

    start_epoch = 1
    start_topic = 0
    shuffle_load = None  # ensures same shuffle for given epoch across checkpoints
    rstate_load = None   # ensures same random state during training
    if(os.path.isfile(os.path.join(args.out_dir, 'cd_event_model_state'))):
        print("Resuming training from last checkpoint...")
        shuffle_load, rstate_load, cd_event_model, cd_event_optimizer, patience_counter, start_epoch, start_topic, event_best_dev_f1 = load_training_checkpoint(
            cd_event_model, cd_event_optimizer, os.path.join(args.out_dir, 'cd_event_model_state'), device)
        shuffle_load, rstate_load, cd_entity_model, cd_entity_optimizer, patience_counter, start_epoch,  start_topic, entity_best_dev_f1 = load_training_checkpoint(
            cd_entity_model, cd_entity_optimizer, os.path.join(args.out_dir, 'cd_entity_model_state'), device)

    cd_event_model = cd_event_model.to(device)
    cd_entity_model = cd_entity_model.to(device)

    torch.cuda.empty_cache()
    orig_event_th = config_dict["event_merge_threshold"]
    orig_entity_th = config_dict["entity_merge_threshold"]
    for epoch in range(start_epoch, config_dict["epochs"]):
        logging.info('Epoch {}:'.format(str(epoch)))
        print(f"Epoch {epoch}:")
        topics_keys = list(topics.keys())

        # allow resuming from partial checkpoints
        if shuffle_load is None:
            if rstate_load is not None:
                random.setstate(rstate_load)
            shuffle_save = random.getstate()
        else:
            # ensures shuffle order is preserved st topic_counter makes sense
            random.setstate(shuffle_load)
            shuffle_save = shuffle_load
            shuffle_load = None

        random.shuffle(topics_keys)
        topics_counter = start_topic
        topics_keys = topics_keys[topics_counter:]

        for topic_id in topics_keys:
            if rstate_load is not None:
                random.setstate(rstate_load)
                rstate_load = None

            topics_counter += 1
            topic = topics[topic_id]

            logging.info("="*73)
            logging.info(f"Topic {topic_id}")
            print(f"\nTopic {topic_id}")

            # init event and entity clusters
            event_mentions, entity_mentions = MU.topic_to_mention_list(topic, is_gold=True)

            if config_dict["train_init_wd_entity_with_gold"]:
                # initialize entity clusters with gold within document entity coreference chains
                wd_entity_clusters = MU.create_gold_wd_clusters_organized_by_doc(entity_mentions, is_event=False)
            else:
                # initialize entity clusters with within document entity coreference system output
                wd_entity_clusters = MU.init_entity_wd_clusters(entity_mentions, doc_to_entity_mentions)

            entity_clusters = []
            for doc_id, clusters in wd_entity_clusters.items():
                entity_clusters.extend(clusters)

            event_clusters = MU.init_cd(event_mentions, is_event=True)

            # initialize cluster representation
            MU.update_lexical_vectors(entity_clusters, cd_entity_model, device,
                                      is_event=False, requires_grad=False)
            MU.update_lexical_vectors(event_clusters, cd_event_model, device,
                                      is_event=True, requires_grad=False)

            entity_th = config_dict["entity_merge_threshold"]
            event_th = config_dict["event_merge_threshold"]

            for i in range(1, config_dict["merge_iters"]+1):
                print()
                logging.info(f"\nIteration number {i}")

                # Entities
                print("Train entity model and merge entity clusters...")
                logging.info("Train entity model and merge entity clusters...")
                train_and_merge(clusters=entity_clusters, other_clusters=event_clusters,
                                model=cd_entity_model, optimizer=cd_entity_optimizer,
                                loss=cd_entity_loss,device=device,topic=topic,is_event=False,epoch=epoch,
                                topics_counter=topics_counter, topics_num=topics_num,
                                threshold=entity_th)
                # Events
                print("\nTrain event model and merge event clusters...")
                logging.info("Train event model and merge event clusters...")
                train_and_merge(clusters=event_clusters, other_clusters=entity_clusters,
                                model=cd_event_model, optimizer=cd_event_optimizer,
                                loss=cd_event_loss,device=device,topic=topic,is_event=True,epoch=epoch,
                                topics_counter=topics_counter, topics_num=topics_num,
                                threshold=event_th)

            if config_dict["save_every_topic"]:
                print(f"\nSaving partial checkpoint for epoch {epoch}, topic {topic_id}")
                rstate_save = random.getstate()
                save_training_checkpoint(shuffle_save, rstate_save, patience_counter, epoch, topics_counter, cd_event_model, cd_event_optimizer, event_best_dev_f1,
                                         filename=os.path.join(args.out_dir, 'cd_event_model_state'))
                save_training_checkpoint(shuffle_save, rstate_save, patience_counter, epoch, topics_counter, cd_entity_model, cd_entity_optimizer, entity_best_dev_f1,
                                         filename=os.path.join(args.out_dir, 'cd_entity_model_state'))

        print("\nTesting models on dev set...")
        logging.info("Testing models on dev set...")

        torch.cuda.empty_cache()

        threshold_list = config_dict["dev_th_range"]
        improved = False
        best_event_f1_for_th = 0
        best_entity_f1_for_th = 0

        if event_best_dev_f1 > 0:
            best_saved_cd_event_model = MU.load_check_point(os.path.join(args.out_dir,
                                                                      'cd_event_best_model'))
            # best_saved_cd_event_model.to(device)
        else:
            best_saved_cd_event_model = cd_event_model

        if entity_best_dev_f1 > 0:
            best_saved_cd_entity_model = MU.load_check_point(os.path.join(args.out_dir,
                                                                       'cd_entity_best_model'))
            # best_saved_cd_entity_model.to(device)
        else:
            best_saved_cd_entity_model = cd_entity_model

        event_threshold = config_dict["event_merge_threshold"]
        entity_threshold = config_dict["entity_merge_threshold"]
        print("Testing models on dev set".format((event_threshold, entity_threshold)))
        logging.info("Testing models on dev set".format((event_threshold, entity_threshold)))

        # test event coref on dev
        event_f1, _ = MU.test_models(dev_set, cd_event_model, best_saved_cd_entity_model, device,
                                  write_clusters=False, out_dir=args.out_dir, config_dict=config_dict,
                                  doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False, epoch=epoch)

        # test entity coref on dev
        _, entity_f1 = MU.test_models(dev_set, best_saved_cd_event_model, cd_entity_model, device,
                                   write_clusters=False, out_dir=args.out_dir, config_dict=config_dict,
                                   doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False, epoch=epoch)

        del best_saved_cd_event_model
        del best_saved_cd_entity_model
        torch.cuda.empty_cache()

        save_epoch_f1(event_f1, entity_f1, epoch, 0.5, 0.5)

        improved = False
        if event_f1 > event_best_dev_f1:
            event_best_dev_f1 = event_f1
            best_event_epoch = epoch
            MU.save_check_point(cd_event_model, os.path.join(args.out_dir, "cd_event_best_model"))
            improved = True
            patience_counter = 0
        if entity_f1 > entity_best_dev_f1:
            entity_best_dev_f1 = entity_f1
            best_entity_epoch = epoch
            MU.save_check_point(cd_entity_model, os.path.join(args.out_dir, "cd_entity_best_model"))
            improved = True
            patience_counter = 0

        if not improved:
            patience_counter += 1

        start_topic = 0
        rstate_save = random.getstate()
        save_training_checkpoint(None, rstate_save, patience_counter, epoch+1, start_topic, cd_event_model, cd_event_optimizer, event_best_dev_f1,
                                 filename=os.path.join(args.out_dir, "cd_event_model_state"))
        save_training_checkpoint(None, rstate_save, patience_counter, epoch+1, start_topic, cd_entity_model, cd_entity_optimizer, entity_best_dev_f1,
                                 filename=os.path.join(args.out_dir, "cd_entity_model_state"))

        logging.info(f"patience: {patience_counter}")
        print(f"Patience: {patience_counter}")
        if patience_counter >= config_dict["patience"]:
            logging.info('Early Stopping!')
            print('Early Stopping!\n')
            save_summary(event_best_dev_f1, entity_best_dev_f1, best_event_epoch, best_entity_epoch, epoch)
            break


def train_and_merge(clusters, other_clusters, model, optimizer,
                    loss, device, topic ,is_event, epoch,
                    topics_counter, topics_num, threshold):
    '''
    This function trains event/entity and then uses agglomerative clustering algorithm that
    merges event/entity clusters
    :param clusters: current event/entity clusters
    :param other_clusters: should be the event current clusters if clusters = entity clusters
    and vice versa.
    :param model: event/entity model (according to clusters parameter)
    :param optimizer: event/entity optimizer (according to clusters parameter)
    :param loss: event/entity loss (according to clusters parameter)
    :param device: gpu/cpu Pytorch device
    :param topic: Topic object represents the current topic
    :param is_event: whether to currently handle event mentions or entity mentions
    :param epoch: current epoch number
    :param topics_counter: the number of current topic
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :return:
    '''

    # Update arguments/predicates vectors according to the other clusters state
    MU.update_args_feature_vectors(clusters, other_clusters, model, device, is_event)

    cluster_pairs, test_cluster_pairs \
        = MU.generate_cluster_pairs(clusters, is_train=True)

    # Train pairwise event/entity coreference scorer
    MU.train(cluster_pairs, model, optimizer, loss, device, topic.docs,
             epoch, topics_counter, topics_num, is_event,
             other_clusters, config_dict)

    with torch.no_grad():
        MU.update_lexical_vectors(clusters, model, device, is_event, requires_grad=False)

        event_mentions, entity_mentions = MU.topic_to_mention_list(topic, is_gold=True)

        # Update span representations after training
        MU.create_mention_span_representations(event_mentions, model, device, topic.docs, is_event=True,
                                               requires_grad=False)
        MU.create_mention_span_representations(entity_mentions, model, device, topic.docs, is_event=False,
                                               requires_grad=False)

        cluster_pairs = test_cluster_pairs

        # Merge clusters till reaching the threshold
        MU.merge(clusters, cluster_pairs, other_clusters, model, device, topic.docs, epoch,
                 topics_counter, topics_num, threshold, is_event, config_dict)


def save_epoch_f1(event_f1, entity_f1, epoch,  best_event_th, best_entity_th):
    '''
    Write to a text file B-cubed F1 measures of both event and entity clustering
    according to the models' predictions on the dev set after each training epoch.
    :param event_f1: B-cubed F1 measure for event coreference
    :param entity_f1: B-cubed F1 measure for entity coreference
    :param epoch: current epoch number
    :param best_event_th: best found merging threshold for event coreference
    :param best_entity_th: best found merging threshold for event coreference
    '''
    with open(os.path.join(args.out_dir, 'epochs_scores.txt'), 'a') as f:
        f.write(f"Epoch {epoch} - Event F1: {event_f1:.4f} with th = {best_event_th} Entity F1: {entity_f1:.4f} with th = {best_entity_th}\n")


def save_summary(best_event_score, best_entity_score, best_event_epoch, best_entity_epoch, total_epochs):
    '''
    Writes to a file a summary of the training (best scores, their epochs, and total number of
    epochs)
    :param best_event_score: best event coreference score on the dev set
    :param best_entity_score: best entity coreference score on the dev set
    :param best_event_epoch: the epoch of the best event coreference
    :param best_entity_epoch: the epoch of the best entity coreference
    :param total_epochs: total number of epochs
    '''
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(f"Best Event F1: {best_event_score:.4f} epoch: {best_event_epoch}\n"
                f"Best Entity F1: {best_entity_score:.4f} epoch: "
                f"{best_entity_epoch} \n Training epochs: {total_epochs}")


def save_training_checkpoint(shuffle, rstate, patience_counter, epoch, topic, model, optimizer, best_f1, filename):
    '''
    Saves model's checkpoint after each epoch
    :param epoch: epoch number
    :param model: the model to save
    :param optimizer: Pytorch optimizer
    :param best_f1: the best B-cubed F1 score so far
    :param filename: the filename of the checkpoint file
    '''
    state = {'shuffle': shuffle, 'rstate': rstate, 'patience_counter': patience_counter,
             'epoch': epoch, 'topic': topic, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'best_f1': best_f1}
    torch.save(state, filename)


def load_training_checkpoint(model, optimizer, filename, device):
    '''
    Loads checkpoint from a file
    :param model: an initialized model (CDCorefScorer)
    :param optimizer: new Pytorch optimizer
    :param filename: the checkpoint filename
    :param device: gpu/cpu device
    :return: model, optimizer, epoch, best_f1 loaded from the checkpoint.
    '''
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    shuffle = checkpoint['shuffle']
    rstate = checkpoint['rstate']
    start_epoch = checkpoint['epoch']
    topic = checkpoint['topic']
    patience_counter = checkpoint['patience_counter']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_f1 = checkpoint['best_f1']
    print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return shuffle, rstate, model, optimizer, patience_counter, start_epoch, topic, best_f1


def main():
    '''
    This script loads the train and dev sets, initializes models, optimizers and loss functions,
    then, it runs the training procedure that alternates between entity and event training and
    their clustering.
    After each epoch, it runs the inference procedure on the dev set, calculates
    the B-cubed measure and use it to tune the model and its hyper-parameters.
    Finally, it saves the entity and event models that achieved the best B-cubed scores
    on the dev set.
    '''
    import shared.classes
    sys.modules['classes'] = shared.classes

    logging.info('Loading training and dev data...')
    with open(config_dict["train_path"], 'rb') as f:
        training_data = pickle.load(f)
    with open(config_dict["dev_path"], 'rb') as f:
        dev_data = pickle.load(f)

    del sys.modules['classes']
    del shared.classes

    logging.info('Training and dev data have been loaded.')

    train_model(training_data, dev_data)


if __name__ == '__main__':
    main()
