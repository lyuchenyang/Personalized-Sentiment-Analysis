""" running training and evaluation code for document-level sentiment analysis project

    Created by Chenyang Lyu
"""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss

import argparse
import sklearn.metrics as metric
import glob
import logging
import os
import random
import numpy as np
import json
import pickle

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertModel,
    BertTokenizer,
    BertConfig,
    BertPreTrainedModel,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaPreTrainedModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling import CrossContextBert, FocalLoss
from utils import eval_to_file, load_data_individual, assign_lr_to_parameters
from pargs import Arguments

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
label_list = ["1", "2", "3", "4", "5"]
label_list_imdb = [str(i) for i in range(1, 11)]

MODEL_CLASSES = {
    "bert-base-uncased": (BertConfig, BertTokenizer, BertModel),
    "bert-large-uncased": (BertConfig, BertTokenizer, BertModel),
    "spanbert-base": (BertConfig, BertTokenizer, BertModel),
    "spanbert-large": (BertConfig, BertTokenizer, BertModel),
    "longformer": (LongformerConfig, LongformerTokenizer, LongformerModel),
    "longformer-large": (LongformerConfig, LongformerTokenizer, LongformerModel)
}

MODEL_NAMES = {
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-uncased": "bert-large-uncased",
    "spanbert-base": "SpanBERT/spanbert-base-cased",
    "spanbert-large": "SpanBERT/spanbert-large-cased",
    "longformer": "allenai/longformer-base-4096",
    "longformer-large": "allenai/longformer-large-4096"
}

args = Arguments('document-level-sa')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_from_bert_embeddings(args, train_dataset, model):
    cache_dir = args.data_dir + args.task_name + '_' + args.model_type + '_user_pord_embed.cache'

    if os.path.isfile(cache_dir):
        user_embed, prod_embed = pickle.load(open(cache_dir, 'rb'))
        model.user_textual_embedding.weight = torch.nn.Parameter(torch.tensor(user_embed, device='cuda'))
        model.product_textual_embedding.weight = torch.nn.Parameter(torch.tensor(prod_embed, device='cuda'))
    else:
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

        user_dict, prod_dict = {}, {}
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                with torch.no_grad():
                    model.eval()
                    batch = tuple(t.to(args.device) for t in batch)

                    user, product = list(batch[4].view(-1).cpu().numpy()), list(batch[5].view(-1).cpu().numpy())
                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                    outputs = model.bert(**inputs)

                    last_hidden_states = outputs[0]
                    attn_mask = inputs['attention_mask']
                    attn_mask[:, 0] = 0

                    attns = attn_mask.unsqueeze(-1).repeat(1, 1, args.hidden_size)

                    masked_outputs = attns * last_hidden_states
                    summed_outputs = torch.sum(masked_outputs, dim=1).cpu().numpy()

                    attn_mask = attn_mask.cpu().numpy()
                    for u, p, am, so in zip(user, product, attn_mask, summed_outputs):
                        fac = np.sum(am, axis=-1)
                        if u not in user_dict:
                            user_dict[u] = []
                        if p not in prod_dict:
                            prod_dict[p] = []

                        norm_inp = so / fac
                        user_dict[u].append(norm_inp)
                        prod_dict[p].append(norm_inp)
            break

        for u in user_dict:
            fac = len(user_dict[u])
            u_ind = torch.tensor(u, dtype=torch.long).unsqueeze(0).cuda()
            all_us = np.array(user_dict[u])
            new_user_embed = torch.tensor(np.sum(all_us, axis=0) / fac).squeeze().unsqueeze(0).cuda()
            with torch.no_grad():
                model.user_textual_embedding.weight.index_copy_(0, u_ind, new_user_embed)

        for p in prod_dict:
            fac = len(prod_dict[p])
            p_ind = torch.tensor(p, dtype=torch.long).unsqueeze(0).cuda()
            new_prod_embed = torch.tensor(np.sum(np.array(prod_dict[p]), axis=0) / fac).squeeze().unsqueeze(0).cuda()
            with torch.no_grad():
                model.product_textual_embedding.weight.index_copy_(0, p_ind, new_prod_embed)
        user_embed, prod_embed = model.user_textual_embedding.weight.detach().cpu().numpy(), \
                                 model.product_textual_embedding.weight.detach().cpu().numpy()
        pickle.dump([user_embed, prod_embed], open(cache_dir, "wb"), protocol=4)

    '''
    The ratio of the F-norm of randomly initialized user/product embedding matrix and the textual user/product embedding
    '''
    u_ratio_fro = torch.linalg.matrix_norm(model.user_embedding.weight) / torch.linalg.matrix_norm(
        model.user_textual_embedding.weight)
    p_ratio_fro = torch.linalg.matrix_norm(model.product_embedding.weight) / torch.linalg.matrix_norm(
        model.product_textual_embedding.weight)

    print(u_ratio_fro)
    print(p_ratio_fro)

    with torch.no_grad():
        model.user_embedding.weight = torch.nn.Parameter(u_ratio_fro*model.user_textual_embedding.weight)
        model.product_embedding.weight = torch.nn.Parameter(p_ratio_fro*model.product_textual_embedding.weight)

    return


def train(args, train_dataset, model, tokenizer, dev_set=None, eval_set=None):
    if args.is_text_embed:
        init_from_bert_embeddings(args, train_dataset, model)
    # exit()
    """ Training the model """
    tb_writer = SummaryWriter()

    num_labels = len(args.label_list)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = (len(train_dataloader) * args.num_train_epochs) // args.gradient_accumulation_steps

    # Prepare optimizer for training
    if args.is_cross_context:
        optimizer_group_parameters = assign_lr_to_parameters(args, model)
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

    optimizer = AdamW(optimizer_group_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_steps * t_total),
                                                num_training_steps=t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     epochs_trained = global_step // len(train_dataloader // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
    #
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    loss_fct = CrossEntropyLoss()

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            user, product = batch[4].view(-1, 1), batch[5].view(-1, 1)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type not in [
                    'longformer'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

            if args.is_cross_context:
                logits, *_ = model(inputs, user, product)
            else:
                inputs['labels'] = batch[3]
                outputs = model(**inputs)
                logits = outputs[1]  # model outputs are always tuple in transformers (see doc)

            loss = loss_fct(logits.view(-1, num_labels), batch[3].view(-1))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if global_step % (500 * args.logging_steps) == 0 and dev_set is not None:
                        # results = evaluate(args, model, dev_set, tokenizer)
                        results = evaluate(args, model, eval_set, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_last_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

    tb_writer.close()
    global_step = 1 if global_step == 0 else global_step

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.train_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            user, product = batch[4].view(-1, 1), batch[5].view(-1, 1)

            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                if args.model_type != 'roberta':
                    inputs['token_type_ids'] = batch[2] if args.model_type not in [
                        'longformer'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                if args.is_cross_context:
                    logits, *_ = model(inputs, user, product)
                else:
                    inputs['labels'] = batch[3]
                    outputs = model(**inputs)
                    logits = outputs[1]  # model outputs are always tuple in transformers (see doc)
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

        preds_logits = {'logits': preds}
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = {'acc': (preds == out_label_ids).mean()}
        results.update(result)

        eval_to_file(args.eval_out_file, preds, out_label_ids, preds_logits)
        print("accuracy: ", metric.accuracy_score(out_label_ids, preds))
        print("precision: ", metric.precision_score(out_label_ids, preds, average='macro'))
        print("recall: ", metric.recall_score(out_label_ids, preds, average='macro'))
        print("F1: ", metric.f1_score(out_label_ids, preds, average='macro'))
        print("Mean Squared Error: ", metric.mean_squared_error(out_label_ids, preds))
        print("Root Mean Squared Error: ", metric.mean_squared_error(out_label_ids, preds, squared=False))

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", type=str, default="yelp-2013",
                        help="the name of the training task (the dataset name)")
    parser.add_argument("--model_size", type=str, default="base",
                        help="the size of pre-trained model")
    parser.add_argument("--model_type", type=str, default="bert-base-uncased",
                        help="the type of pre-trained model")
    parser.add_argument("--epochs", type=int, default=2,
                        help="the numebr of training epochs")
    parser.add_argument("--cross_context", action="store_true",
                        help="use cross_context mode")
    parser.add_argument("--text_embed", action="store_true",
                        help="use historical reviews to initialize user and product emebddings")
    parser.add_argument("--do_train", action="store_true",
                        help="whether to train the model or not")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether to evaluate the model or not")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="the weight decay rate")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="the learning rate used to train the model")
    parser.add_argument("--warmup_steps", type=float, default=0.0,
                        help="the warm_up step rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="the maximum sequence length used to load dataset")
    parser.add_argument("--seed", type=int, default=0,
                        help="the random seed used in model initialization and dataloader")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="the batch size used in training and evaluation")
    parser.add_argument("--logging_steps", type=int, default=400,
                        help="the warm_up step rate")
    parser.add_argument("--device", type=int, default=0,
                        help="the device id used for training and evaluation")
    parser.add_argument("--user_emb_size", type=int, default=768,
                        help="the user embedding size")
    parser.add_argument("--product_emb_size", type=int, default=768,
                        help="the product embedding size")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="the attention heads used in multi head attention function")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")

    arguments, _ = parser.parse_known_args()

    args.task_name = arguments.task_name
    args.model_size = arguments.model_size
    args.num_train_epochs = arguments.epochs
    args.is_cross_context = arguments.cross_context
    args.is_text_embed = arguments.text_embed
    args.do_train = arguments.do_train
    args.do_eval = arguments.do_eval
    args.weight_decay = arguments.weight_decay
    args.learning_rate = arguments.learning_rate
    args.warmup_steps = arguments.warmup_steps
    args.max_seq_length = arguments.max_seq_length
    args.seed = arguments.seed
    args.train_batch_size = arguments.batch_size
    args.device = torch.device("cuda:" + str(arguments.device))
    args.model_type = arguments.model_type
    args.logging_steps = arguments.logging_steps
    args.gradient_accumulation_steps = arguments.gradient_accumulation_steps
    args.data_dir = 'data/document-level-sa-dataset/{}/'.format(args.task_name)

    args.do_train = True
    args.is_cross_context = True

    if args.task_name == 'imdb':
        args.label_list = label_list_imdb
    else:
        args.label_list = label_list

    # args.model_type = "longformer"
    args.model_name_or_path = MODEL_NAMES[args.model_type]

    if args.is_cross_context:
        model_type = "cross_context"
    else:
        model_type = 'vanilla'

    output_dir = "trained_models/" + args.model_type + "_" + args.task_name + '_' + model_type + "_epochs_" + str(
        args.num_train_epochs) + "_lr_" + \
                 str(args.learning_rate) + "_weight-decay_" + str(args.weight_decay) + "_warmup_" + str(
        args.warmup_steps) + "_mql_" + str(args.max_seq_length) + \
                 "_seed_" + str(args.seed) + "/"
    eval_dir = "eval_results/" + args.model_type + "_" + args.task_name + '_' + model_type + "_epochs_" + str(
        args.num_train_epochs) + "_lr_" + \
               str(args.learning_rate) + "_weight-decay_" + str(args.weight_decay) + "_warmup_" + str(
        args.warmup_steps) + "_mql_" + str(args.max_seq_length) + \
               "_seed_" + str(args.seed) + ".log"

    args.output_dir = output_dir
    args.eval_out_file = eval_dir

    config_class, tokenizer_class, model_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
    )
    num_labels = len(args.label_list)

    data_dirs = ["data/document-level-sa-dataset/{}/".format(args.task_name) + args.task_name + "-seg-20-20.train.ss",
                 "data/document-level-sa-dataset/{}/".format(args.task_name) + args.task_name + "-seg-20-20.dev.ss",
                 "data/document-level-sa-dataset/{}/".format(args.task_name) + args.task_name + "-seg-20-20.test.ss",
                 ]

    set_seed(args)
    args.model_type = args.model_type.lower()
    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
    )
    config.output_attention = True
    config.output_hidden_states = True

    # config.user_emb_size = arguments.user_emb_size
    # config.product_emb_size = arguments.product_emb_size
    config.user_emb_size = config.hidden_size
    config.product_emb_size = config.hidden_size

    config.attention_heads = arguments.attention_heads
    config.max_seq_length = arguments.max_seq_length
    config.model_type = args.model_type
    args.hidden_size = config.hidden_size

    train_dataset, dev_dataset, test_dataset, user_vocab, prod_vocab = load_data_individual(args, data_dirs, tokenizer)

    if args.is_cross_context:
        if 'longformer' not in config.model_type:
            model_class = CrossContextBert
            model = model_class.from_pretrained(
                args.model_name_or_path,
                u_num_embeddings=user_vocab.word_count,
                p_num_embeddings=prod_vocab.word_count,
                config=config,
            )
        else:
            model = CrossContextBert(
                u_num_embeddings=user_vocab.word_count,
                p_num_embeddings=prod_vocab.word_count,
                config=config)
            model.bert = model_class.from_pretrained(args.model_name_or_path)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # print(args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, dev_set=dev_dataset, eval_set=test_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            if args.is_cross_context:
                model_class = CrossContextBert
                model = model_class.from_pretrained(
                    checkpoint,
                    u_num_embeddings=user_vocab.word_count,
                    p_num_embeddings=prod_vocab.word_count,
                    config=config,
                )
            else:
                model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, dev_dataset, tokenizer, prefix=prefix)
            result = evaluate(args, model, test_dataset, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == '__main__':
    main()
