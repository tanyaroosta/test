#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import pandas as pd
import pickle
import numpy as np
import torch
import re
import math

import torch.nn.functional as F
import os
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
from numpy.random import RandomState
import tqdm

import datasets
import nltk  # Here to have a nice missing dependency error message early on
from datasets import load_dataset, load_metric

from accelerate import Accelerator, DistributedType
from torch.utils.data import DataLoader

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    SchedulerType,
    get_scheduler,
    AdamW,
    get_linear_schedule_with_warmup,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import warnings
warnings.filterwarnings('ignore')


def main():
    gradient_accumulation_steps=1
    num_train_epochs=3
    per_device_train_batch_size=8
    per_device_eval_batch_size=8
    source_prefix="summarize: " 
    do_train=True
    do_eval=True
    ignore_pad_token_for_loss = True
    preprocessing_num_workers = None   # could be an int if on GPU
    max_source_length=1024
    resize_position_embeddings = None
    val_max_target_length = None
    num_beam = None

    max_train_samples=15909

    model_name="sshleifer/distilbart-xsum-12-6"
    train_file='/home/ec2-user/SageMaker/SQ/train.csv'
    test_file='/home/ec2-user/SageMaker/SQ/test.csv'
    output_dir='/home/ec2-user/SageMaker/SQ/'
    
    config = {"lr": 2e-5, "num_epochs": 3, "correct_bias": True, "seed": 42, "batch_size": 16}
    
    torch.cuda.empty_cache()
    
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    correct_bias = config["correct_bias"]
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    accelerator = Accelerator()
    set_seed(seed)

    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, OSError):
        if is_offline_mode():
            raise LookupError(
                "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
            )
        with FileLock(".lock") as lock:
            nltk.download("punkt", quiet=True)


    summarization_name_mapping = {
        "amazon_reviews_multi": ("review_body", "review_title"),
        "big_patent": ("description", "abstract"),
        "cnn_dailymail": ("article", "highlights"),
        "orange_sum": ("text", "summary"),
        "pn_summary": ("article", "summary"),
        "psc": ("extract_text", "summary_text"),
        "samsum": ("dialogue", "summary"),
        "thaisum": ("body", "summary"),
        "xglue": ("news_body", "news_title"),
        "xsum": ("document", "summary"),
        "wiki_summary": ("article", "highlights"),
    }


    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    max_predict_samples=None
    
    data_files = {}
    
    data_files["train"] = train_file
    extension = train_file.split(".")[-1]

    data_files["test"] = test_file
    extension = test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
   
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_name,
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        config=config,
        use_auth_token=None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < max_source_length
    ):
        if resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {max_source_length}."
            )
            model.resize_position_embeddings(max_source_length)
        elif resize_position_embeddings:
            model.resize_position_embeddings(max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = source_prefix

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if do_train:
        column_names = raw_datasets["train"].column_names
    elif do_eval:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(None)
    text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
   
    summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
   

    # Temporarily set max_target_length for training.
    max_target_length = 128
    padding = "max_length" 
    max_seq_length = tokenizer.model_max_length

    
    def preprocess_function(examples):

        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        
    with accelerator.main_process_first():     
        train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=column_names,
            )
        
     # Data collator
    label_pad_token_id = -100 

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

        

    if do_eval:
        max_target_length = val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["test"]
        
        with accelerator.main_process_first():        
            eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc= preprocessing_num_workers,
                    remove_columns=column_names,
                )
            
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size)    

   
    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn= data_collator, batch_size=per_device_train_batch_size
    )
    

    # Optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)
    
    # Metric
    metric = load_metric("rouge")
    
    model = model.to(accelerator.device)
    
    # prepare model
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=max_train_steps,
    )
    
    print('available"',torch.cuda.is_available())
    print('current device:',torch.cuda.current_device())
    print('device count:',torch.cuda.device_count())
    print('device name:',torch.cuda.get_device_name(torch.cuda.current_device()))
    print('Allocated:', round(torch.cuda.memory_allocated(torch.cuda.current_device())/1024**3,1), 'GB')
    print('accelerator device"',accelerator.device)
    
    
    # Train!
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm.tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

#         model.eval()
#         losses = []
#         for step, batch in enumerate(eval_dataloader):
#             # We could avoid this line since we set the accelerator with `device_placement=True`.
#             batch.to(accelerator.device)
#             with torch.no_grad():
#                 outputs = model(**batch)

#             loss = outputs.loss
#             losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

#         losses = torch.cat(losses)
#         losses = losses[: len(eval_dataset)]
    
    
    

if __name__ == "__main__":
    main()

