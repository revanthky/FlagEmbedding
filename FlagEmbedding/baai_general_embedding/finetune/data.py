## 10/14/24 (October 14th): updated the flagembedding code, to use MNR Loss 
# main change: "passage" variable is divided into positives and negatives
# currently not batching the negatives, is this necessary? using as a list directly in the model forward pass

import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments


# class TrainDatasetForEmbedding(Dataset):
#     def __init__(
#             self,
#             args: DataArguments,
#             tokenizer: PreTrainedTokenizer
#     ):
#         if os.path.isdir(args.train_data):
#             train_datasets = []
#             for file in os.listdir(args.train_data):
#                 temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
#                                                      split='train')
#                 if len(temp_dataset) > args.max_example_num_per_dataset:
#                     temp_dataset = temp_dataset.select(
#                         random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
#                 train_datasets.append(temp_dataset)
#             self.dataset = datasets.concatenate_datasets(train_datasets)
#         else:
#             self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

#         self.tokenizer = tokenizer
#         self.args = args
#         self.total_len = len(self.dataset)

#     def __len__(self):
#         return self.total_len

#     def __getitem__(self, item) -> Tuple[str, List[str]]:
#         query = self.dataset[item]['query']
#         if self.args.query_instruction_for_retrieval is not None:
#             query = self.args.query_instruction_for_retrieval + query

#         passages = []

#         assert isinstance(self.dataset[item]['pos'], list)
#         pos = random.choice(self.dataset[item]['pos'])
#         passages.append(pos)

#         if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
#             num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
#             negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
#         else:
#             negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
#         passages.extend(negs)

#         if self.args.passage_instruction_for_retrieval is not None:
#             passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
#         return query, passages

class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file), split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset)
                    )
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, str, List[str]]:
        """
        Returns a tuple of query, a single positive passage, and a list of negative passages.
        """
        # Get the query
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval:
            query = self.args.query_instruction_for_retrieval + query

        # Get the positive passage (only one random positive)
        assert isinstance(self.dataset[item]['pos'], list)
        positive = random.choice(self.dataset[item]['pos'])

        # Get the negative passages (ensuring enough negatives)
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negatives = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negatives = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        # Optionally prepend instructions to passages
        if self.args.passage_instruction_for_retrieval:
            positive = self.args.passage_instruction_for_retrieval + positive
            negatives = [self.args.passage_instruction_for_retrieval + neg for neg in negatives]

        return query, positive, negatives  # Return separately
@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that converts List[Tuple[query, positive, negatives]] to 
    a batch of tokenized queries, a batch of tokenized positives, 
    and a batch of tokenized negatives.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        """
        features: List of tuples, where each tuple contains:
                  (query, positive passage, negatives)
                  query: str, positive: str, negatives: List[str]
        """
        queries = [f[0] for f in features]  # Extract queries
        positives = [f[1] for f in features]  # Extract positive passages
        all_negatives = [f[2] for f in features]  # Extract lists of negatives

        # Flatten out the negatives, creating a single list of negatives (RIGHT NOW: NOT USING BATCHED NEGATIVES)
        # flattened_negatives = sum(all_negatives, [])

        # Tokenize the queries
        q_collated = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )

        # Tokenize the positive passages
        p_collated = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )

        neg_collated = [
            self.tokenizer(
                negatives,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
            for negatives in all_negatives
        ]
        # Tokenize the negatives (batched together)
        # neg_collated = self.tokenizer(
        #     flattened_negatives, 
        #     padding=True,
        #     truncation=True,
        #     max_length=self.passage_max_len,
        #     return_tensors="pt",
        # ) #flattened_negs taken out

        return {"query": q_collated, "positive": p_collated, "negatives": neg_collated}
# class EmbedCollator(DataCollatorWithPadding):
#     """
#     Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
#     and pass batch separately to the actual collator.
#     Abstract out data detail for the model.
#     """
#     query_max_len: int = 32
#     passage_max_len: int = 128

#     def padding_score(self, teacher_score):
#         group_size = None
#         for scores in teacher_score:
#             if scores is not None:
#                 group_size = len(scores)
#                 break
#         if group_size is None:
#             return None

#         padding_scores = [100.0] + [0.0] * (group_size - 1)
#         new_teacher_score = []
#         for scores in teacher_score:
#             if scores is None:
#                 new_teacher_score.append(padding_scores)
#             else:
#                 new_teacher_score.append(scores)
#         return new_teacher_score

#     def __call__(self, features):
#         query = [f[0] for f in features]
#         passage = [f[1] for f in features]

#         if isinstance(query[0], list):
#             query = sum(query, [])
#         if isinstance(passage[0], list):
#             passage = sum(passage, [])

#         q_collated = self.tokenizer(
#             query,
#             padding=True,
#             truncation=True,
#             max_length=self.query_max_len,
#             return_tensors="pt",
#         )
#         d_collated = self.tokenizer(
#             passage,
#             padding=True,
#             truncation=True,
#             max_length=self.passage_max_len,
#             return_tensors="pt",
#         )
#         return {"query": q_collated, "passage": d_collated}

