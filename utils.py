import sklearn.metrics as metric
import json
import torch
from transformers import InputExample
from tqdm import tqdm
from torch.utils.data import TensorDataset
import numpy as np
import os
import pickle
import pandas as pd
import codecs
from collections import OrderedDict
from functools import partial
from time import time

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker


from sklearn import manifold
import numpy as np

import logging
from transformers import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)
json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def convert_examples_to_features(args, examples, tokenizer, max_length=512, label_list=None, output_mode=None,
                                 pad_on_left=False, pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    over_len = 0
    print(args.model_type, args.model_mode)
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        encoded_text = tokenizer.encode(example.text_a)
        if len(encoded_text) > max_length:
            over_len += 1
            if 'longformer' in args.model_type and max_length != 512:
                # if 'longformer' in args.model_type or 'incremental' in args.model_mode or 'baseline' in args.model_mode:
                input_ids = encoded_text[:max_length]
            else:
                input_ids = encoded_text[:129] + encoded_text[-383:]
            token_type_ids = [0] * max_length
        else:
            inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True,
                                           max_length=max_length, )

            if 'longformer' in args.model_type:
                input_ids = inputs["input_ids"]
                token_type_ids = [0] * len(input_ids)
            else:
                input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    print('over_len: ', float(over_len / len(examples)))
    return features


def assign_lr_to_parameters(args, model):
    if args.is_cross_context:
        optimizer_group_parameters = [
            # {
            #     "params": [p for n, p in model.named_parameters() if 'bert' in n],
            #     "lr": args.learning_rate,
            # },
            {
                # "names": [n for n, p in model.named_parameters() if 'bert' not in n],
                "params": [p for n, p in model.named_parameters() if 'bert' not in n],
                "lr": 2 * args.learning_rate,
            }
        ]
        layers = 24 if 'large' in args.model_type else 12
        defactor = 0.90
        for i in range(layers):
            layer_i = 'layer.' + str(i) + '.'
            optimizer_group_parameters.append({
                # "names": [n for n, p in model.named_parameters() if 'bert' in n and layer_i in n],
                "params": [p for n, p in model.named_parameters() if 'bert' in n and layer_i in n],
                "lr": args.learning_rate * pow(defactor, layers - i),
            })
        optimizer_group_parameters.append({
            # "names": [n for n, p in model.named_parameters() if 'bert.embeddings' in n],
            "params": [p for n, p in model.named_parameters() if 'bert.embeddings' in n],
            "lr": args.learning_rate * pow(defactor, layers),
        })
        optimizer_group_parameters.append({
            # "names": [n for n, p in model.named_parameters() if 'bert.pooler' in n],
            "params": [p for n, p in model.named_parameters() if 'bert.pooler' in n],
            "lr": args.learning_rate,
        })
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_group_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

    return optimizer_group_parameters


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<unk>': 0}
        self.word2count = {}
        self.index2word = {}
        self.word_count = 1

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.word_count
            self.index2word[self.word_count] = word
            self.word_count += 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)


def eval_to_file(out_file_name, preds, true_labels, preds_logits='None'):
    accuracy = metric.accuracy_score(true_labels, preds)
    precision = metric.precision_score(true_labels, preds, average='macro')
    recall = metric.recall_score(true_labels, preds, average='macro')
    f1 = metric.f1_score(true_labels, preds, average='macro')
    MSE = metric.mean_squared_error(true_labels, preds)
    RMSE = metric.mean_squared_error(true_labels, preds, squared=False)

    model_statistics = {}
    line = "accuracy: " + str(accuracy) + "\n" + "precision: " + str(precision) + "\n" + "recall: " + str(
        recall) + "\n" + "F1: " + str(f1) + "\n" + "MSE: " + str(MSE) + "\n" + "RMSE: " + str(RMSE) + "\n"
    model_statistics['statistics'] = line
    model_statistics['preds'] = preds
    model_statistics['ground_truth'] = true_labels
    # model_statistics['preds_logits'] = preds_logits['logits']

    # ms = json.dumps(model_statistics)
    # json_dump([model_statistics], out_file_name)
    pickle.dump([model_statistics], open(out_file_name, "wb"), protocol=4)


def remove_chars(text, target):
    for t in target:
        text = text.replace(t, "")
    return text


def build_vocab(dirs):
    user_vocab = Lang('user')
    prod_vocab = Lang('product')

    for dir in tqdm(dirs, desc="Building vocab"):
        df = pd.read_csv(dir)
        users, prods = list(df['user']), list(df['product'])

        for i, (u, p) in enumerate(tqdm(zip(users, prods))):
            a, b = u, p
            user_vocab.addWord(a)
            prod_vocab.addWord(b)

    return user_vocab, prod_vocab


def load_data_individual(args, data_dirs, tokenizer):
    user_vocab, prod_vocab = build_vocab(data_dirs)
    datasets = []
    target = ['<sssss>']
    for dir in tqdm(data_dirs, desc="Loading dataset"):
        cache_dir = dir + ".cache_" + args.model_type + '_' + args.model_mode + str(args.max_seq_length)
        # cache_dir = dir + ".cache_bert-base-uncased"

        if os.path.isfile(cache_dir):
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_user_ids, all_product_ids, user_vocab, product_vocab = pickle.load(
                open(cache_dir, 'rb'))
        else:
            examples = []
            user_ids, product_ids = [], []

            df = pd.read_csv(dir)
            users, prods, labels, texts = list(df['user']), list(df['product']), \
                                          list(df['label']), list(df['text']),

            for i, (u, p, label, text) in enumerate(tqdm(zip(users, prods, labels, texts))):
                guid = "%s-%s" % ("document-sa", i)
                a, b, c, d = u, p, label, text
                text_a = remove_chars(d, target)

                label = str(c)

                user_ids.append(user_vocab.word2index[a])
                product_ids.append(prod_vocab.word2index[b])
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
                )
            features = convert_examples_to_features(
                args,
                examples,
                tokenizer,
                max_length=args.max_seq_length,
                label_list=args.label_list,
                output_mode=args.output_mode,
            )
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            if args.output_mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            if args.output_mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
            all_user_ids = torch.tensor(user_ids, dtype=torch.long)
            all_product_ids = torch.tensor(product_ids, dtype=torch.long)

            pickle.dump(
                [all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_user_ids, all_product_ids,
                 user_vocab, prod_vocab], open(cache_dir, "wb"), protocol=4)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_user_ids,
                                all_product_ids)

        datasets.append(dataset)
    datasets.append(user_vocab)
    datasets.append(prod_vocab)
    return datasets


def get_label_distribution(datadir, reverse=True):
    sta = {}
    with open(datadir, " r") as f:
        lines = f.readlines()

        for line in lines:
            a, b, c, d = line.split('\t\t')
            label = int(c)
            if label not in sta:
                sta[label] = 0
            sta[label] += 1
        total = sum(sta[k] for k in sta)
        distri = [sta[e + 1] / total for e in range(len(sta))]
        if reverse:
            distri = [1 / e for e in distri]
        return distri


def process_user_product_review(dataset_name='imdb'):
    data_dirs = ["data/document-level-sa-dataset/" + dataset_name + "-seg-20-20.train.ss",
                 "data/document-level-sa-dataset/" + dataset_name + "-seg-20-20.dev.ss",
                 "data/document-level-sa-dataset/" + dataset_name + "-seg-20-20.test.ss",
                 ]

    user_vocab = Lang('user')
    prod_vocab = Lang('product')

    for dir in data_dirs:
        df = pd.read_csv(dir)
        users, prods = list(df['user']), list(df['product'])

        for i, (u, p) in enumerate(tqdm(zip(users, prods))):
            a, b = u, p
            user_vocab.addWord(a)
            prod_vocab.addWord(b)
        break

    u_not_in = 0
    p_not_in = 0
    for dir in data_dirs:
        df = pd.read_csv(dir)
        users, prods = list(df['user']), list(df['product'])

        for i, (u, p) in enumerate(tqdm(zip(users, prods))):
            a, b = u, p
            if a not in user_vocab.word2index:
                u_not_in += 1
            if b not in prod_vocab.word2index:
                p_not_in += 1
    print(u_not_in, p_not_in)


def draw_samples(lis, ratio):
    samples = ratio if ratio > 1 else int(ratio * len(lis))

    if samples > len(lis):
        new_lis = np.random.choice(len(lis), samples, replace=True)
    else:
        new_lis = np.random.choice(len(lis), samples, replace=False)

    n_lis = [lis[i] for i in new_lis]

    return n_lis


def sample_user_product_reviews():
    task = 'yelp-2013'
    data_dirs = ["data/document-level-sa-dataset/{}/{}-seg-20-20.train.ss".format(task, task),
                 "data/document-level-sa-dataset/{}/{}-seg-20-20.dev.ss".format(task, task),
                 "data/document-level-sa-dataset/{}/{}-seg-20-20.test.ss".format(task, task),
                 ]

    user_dict, prod_dict = {}, {}

    df = pd.read_csv(data_dirs[0])
    users, prods, labels, texts = list(df['user']), list(df['product']), \
                                  list(df['label']), list(df['text']),

    def export_dict(dic, prop, key='prod'):
        lis = []
        for k in dic:
            lis.extend(dic[k])
        print(key, prop, len(lis))

        np.random.shuffle(lis)
        with open('data/baseline_data/{}-seg-20-20.train.ss'.format(task) + '_' + key + '_' + str(prop), 'w') as f:
            sep = ' \t\t '
            for li in lis:
                li = [str(l) for l in li]
                line = li[0] + sep + li[1] + sep + li[2] + sep + li[3] + ' \n'
                f.write(line)

        d = {'user': [li[0] for li in lis],
             'product': [li[1] for li in lis],
             'label': [li[2] for li in lis],
             'text': [li[3] for li in lis]}

        df = pd.DataFrame(d)
        df.to_csv(
            'data/document-level-sa-dataset/{}/{}-seg-20-20.train.ss'.format(task, task) + '_' + key + '_' + str(prop),
            index=False)

    def export_dev_and_test(prop, key='prod'):
        dev_user_dict = {}

        df = pd.read_csv(data_dirs[1])
        users, prods, labels, texts = list(df['user']), list(df['product']), \
                                      list(df['label']), list(df['text']),

        for i, (u, p, label, text) in enumerate(tqdm(zip(users, prods, labels, texts))):
            if u not in dev_user_dict:
                dev_user_dict[u] = []
            dev_user_dict[u].append([u, p, label, text])

        test_user_dict = {}

        df = pd.read_csv(data_dirs[2])
        users, prods, labels, texts = list(df['user']), list(df['product']), \
                                      list(df['label']), list(df['text']),

        for i, (u, p, label, text) in enumerate(tqdm(zip(users, prods, labels, texts))):
            if u not in test_user_dict:
                test_user_dict[u] = []
            test_user_dict[u].append([u, p, label, text])

        def export_(dic, type):
            lis = []
            for k in dic:
                lis.extend(dic[k])

            with open('data/document-level-sa-dataset/original/{}-seg-20-20.{}.ss'.format(task,
                                                                                          type) + '_' + key + '_' + str(
                prop), 'w') as f:
                sep = ' \t\t '
                for li in lis:
                    li = [str(l) for l in li]
                    line = li[0] + sep + li[1] + sep + li[2] + sep + li[3] + ' \n'
                    f.write(line)

            d = {'user': [li[0] for li in lis],
                 'product': [li[1] for li in lis],
                 'label': [li[2] for li in lis],
                 'text': [li[3] for li in lis]}

            df = pd.DataFrame(d)
            df.to_csv(
                'data/document-level-sa-dataset/{}/{}-seg-20-20.{}.ss'.format(task, task, type) + '_' + key + '_' + str(
                    prop),
                index=False)

        export_(dev_user_dict, type='dev')
        export_(test_user_dict, type='test')

    for i, (u, p, label, text) in enumerate(tqdm(zip(users, prods, labels, texts))):
        if u not in user_dict:
            user_dict[u] = []
        user_dict[u].append([u, p, label, text])

        if p not in prod_dict:
            prod_dict[p] = []
        prod_dict[p].append([u, p, label, text])

    for i in range(1, 11):
        prop = 0.1 * i
        new_user_dict, new_prod_dict = {}, {}

        for u in user_dict:
            new_user_dict[u] = draw_samples(user_dict[u], prop)

        for p in prod_dict:
            new_prod_dict[p] = draw_samples(prod_dict[p], prop)

        export_dict(new_user_dict, prop, key='user')
        export_dict(new_prod_dict, prop, key='prod')
        export_dev_and_test(prop, key='user')