#10/14/24: Updated to use MNR loss
# key changes: in the forward pass, but also the input to the forward pass changed (data.py file)
# also, new loss function called in the forward pass (not using cross entropy anymore)
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):#changed
        if features is None:
            return None
        model_output = self.model(**features, return_dict=True)
        embeddings = self.sentence_embedding(model_output.last_hidden_state, features['attention_mask'])
        if self.normlized:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.contiguous()

    def compute_similarity(self, q_reps, p_reps):#changed
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
    

    def multiple_negative_ranking_loss(self, query_reps, positive_reps, negative_reps, margin=0.1):#changed
        pos_scores = torch.matmul(query_reps, positive_reps.T)  # Query-positive similarity
        neg_scores = torch.matmul(query_reps, negative_reps.T)  # Query-negative similarities

        # Margin Ranking Loss
        ranking_loss = torch.nn.functional.margin_ranking_loss(
            pos_scores, neg_scores, target=torch.ones_like(pos_scores), margin=margin
        )
        return ranking_loss
        # not using the compute similarity function at all? is it necessary?

    # def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
    #     q_reps = self.encode(query)
    #     p_reps = self.encode(passage)

    #     if self.training:
    #         if self.negatives_cross_device and self.use_inbatch_neg:
    #             q_reps = self._dist_gather_tensor(q_reps)
    #             p_reps = self._dist_gather_tensor(p_reps)

    #         group_size = p_reps.size(0) // q_reps.size(0)
    #         if self.use_inbatch_neg:
    #             scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
    #             scores = scores.view(q_reps.size(0), -1)

    #             target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
    #             target = target * group_size
    #             loss = self.compute_loss(scores, target)
    #         else:
    #             scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

    #             scores = scores.view(q_reps.size(0), -1)
    #             target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
    #             loss = self.compute_loss(scores, target)

    #     else:
    #         scores = self.compute_similarity(q_reps, p_reps)
    #         loss = None
    #     return EncoderOutput(
    #         loss=loss,
    #         scores=scores,
    #         q_reps=q_reps,
    #         p_reps=p_reps,
    #     )

    def forward(self, query: Dict[str, Tensor], positive: Dict[str, Tensor], negatives: List[Dict[str, Tensor]]): #changed
        """
        Takes tokenized query, positive passage, and negative passages, and computes the loss.
        """
        # Encode query
        q_reps = self.encode(query)

        # Encode positive passage
        positive_reps = self.encode(positive)

        # Encode negative passages
        negative_reps = torch.cat([self.encode(neg) for neg in negatives], dim=0)

        # Compute Multiple Negative Ranking Loss
        loss = self.multiple_negative_ranking_loss(q_reps, positive_reps, negative_reps)

        return EncoderOutput(
            loss=loss,
            q_reps=q_reps,
            p_reps=torch.cat([positive_reps, negative_reps], dim=0)
        )

    # def compute_loss(self, scores, target):
    #     return self.cross_entropy(scores, target)

   

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


# class BiEncoderModel(nn.Module):
#     TRANSFORMER_CLS = AutoModel

#     def __init__(self,
#                  model_name: str = None,
#                  normlized: bool = False,
#                  sentence_pooling_method: str = 'cls',
#                  negatives_cross_device: bool = False,
#                  temperature: float = 1.0,
#                  use_inbatch_neg: bool = True
#                  ):
#         super().__init__()
#         self.model = AutoModel.from_pretrained(model_name)
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

#         self.normlized = normlized
#         self.sentence_pooling_method = sentence_pooling_method
#         self.temperature = temperature
#         self.use_inbatch_neg = use_inbatch_neg
#         self.config = self.model.config

#         if not normlized:
#             self.temperature = 1.0
#             logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
#         if normlized:
#             if self.temperature > 0.5:
#                 raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

#         self.negatives_cross_device = negatives_cross_device
#         if self.negatives_cross_device:
#             if not dist.is_initialized():
#                 raise ValueError('Distributed training has not been initialized for representation all gather.')
#             #     logger.info("Run in a single GPU, set negatives_cross_device=False")
#             #     self.negatives_cross_device = False
#             # else:
#             self.process_rank = dist.get_rank()
#             self.world_size = dist.get_world_size()

#     def gradient_checkpointing_enable(self, **kwargs):
#         self.model.gradient_checkpointing_enable(**kwargs)

#     def sentence_embedding(self, hidden_state, mask):
#         if self.sentence_pooling_method == 'mean':
#             s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
#             d = mask.sum(axis=1, keepdim=True).float()
#             return s / d
#         elif self.sentence_pooling_method == 'cls':
#             return hidden_state[:, 0]

#     def encode(self, features):
#         if features is None:
#             return None
#         psg_out = self.model(**features, return_dict=True)
#         p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
#         if self.normlized:
#             p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
#         return p_reps.contiguous()

#     def compute_similarity(self, q_reps, p_reps):
#         if len(p_reps.size()) == 2:
#             return torch.matmul(q_reps, p_reps.transpose(0, 1))
#         return torch.matmul(q_reps, p_reps.transpose(-2, -1))
    

#     def multiple_negative_ranking_loss(pos_scores,neg_scores, target, margin=0.1):
#         """
#         Compute the multiple negative ranking loss using margin ranking loss.
        
#         query_reps: Tensor, the query embeddings
#         positive_reps: Tensor, the positive passage embeddings
#         negative_reps: Tensor, the negative passage embeddings (multiple negatives)
        
#         margin: float, the margin by which the positive examples should be ranked higher than the negatives.
#         """
#         # Calculate similarity scores
#         # pos_scores = torch.matmul(query_reps, positive_reps.T)  # Similarity with positive example
#         # neg_scores = torch.matmul(query_reps, negative_reps.T)  # Similarity with negative examples
        
#         # The loss ensures that positive scores are greater than negative scores by the margin
#         print(pos_scores)
#         print(pos_scores.shape) # there is no shape attribute
#         print("**"*40)
#         print(neg_scores)
#         print(neg_scores.shape)
#         print(target)
#         print(target.shape)
#         ranking_loss = F.margin_ranking_loss(
#             pos_scores,  # Positive similarity
#             neg_scores,  # Negative similarity
#             target,  # Target (1 because we want pos > neg)
#             margin=margin
#         )
        
#         return ranking_loss

#     # def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
#     #     q_reps = self.encode(query)
#     #     p_reps = self.encode(passage)

#     #     if self.training:
#     #         if self.negatives_cross_device and self.use_inbatch_neg:
#     #             q_reps = self._dist_gather_tensor(q_reps)
#     #             p_reps = self._dist_gather_tensor(p_reps)

#     #         group_size = p_reps.size(0) // q_reps.size(0)
#     #         if self.use_inbatch_neg:
#     #             scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
#     #             scores = scores.view(q_reps.size(0), -1)

#     #             target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
#     #             target = target * group_size
#     #             loss = self.compute_loss(scores, target)
#     #         else:
#     #             scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

#     #             scores = scores.view(q_reps.size(0), -1)
#     #             target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
#     #             loss = self.compute_loss(scores, target)

#     #     else:
#     #         scores = self.compute_similarity(q_reps, p_reps)
#     #         loss = None
#     #     return EncoderOutput(
#     #         loss=loss,
#     #         scores=scores,
#     #         q_reps=q_reps,
#     #         p_reps=p_reps,
#     #     )

#     def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
#         q_reps = self.encode(query)   # Encode the query
#         p_reps = self.encode(passage) # Encode the passages (positive + negatives) SPLIT UP FIRST, THEN ENCODE??)
#         print("passage_variable")
#         print(type(passage))
#         print(p_reps)
#         print(p_reps.shape)
#         print("*"*50)

#         # Split the positive and negative passages
#         positive_reps = p_reps[:1]  # Assuming first passage in batch is the positive
#         print(positive_reps)
#         negative_reps = p_reps[1:]  # Remaining passages are negatives
#         print(negative_reps[0])

#         print(type(q_reps))  # Should print <class 'torch.Tensor'>
#         print(type(positive_reps))  # Should print <class 'torch.Tensor'>
#         print(type(negative_reps))
#         print(negative_reps.size(1))

#         if self.training:
#             # If using multiple negative ranking loss, replace cross-entropy with ranking loss
#             if self.negatives_cross_device and self.use_inbatch_neg:
#                 q_reps = self._dist_gather_tensor(q_reps)
#                 positive_reps = self._dist_gather_tensor(positive_reps)

#             # group_size = p_reps.size(0) // q_reps.size(0)
#             if self.use_inbatch_neg:
#                 # scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
#                 # scores = scores.view(q_reps.size(0), -1)

#                 # target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
#                 # target = target * group_size
#                 # loss = self.compute_loss(scores, target)

#                 #  p_reps.transpose(-2, -1)

#                 pos_scores = torch.matmul(q_reps, positive_reps.transpose(-2, -1))  # Similarity with positive example
#                 neg_scores = torch.matmul(q_reps, negative_reps.transpose(-2, -1))  # Similarity with negative examples
                
#                 pos_scores = pos_scores.view(q_reps.size(0), -1)
#                 neg_scores = neg_scores.view(q_reps.size(0), -1)
#                 target = torch.ones_like(pos_scores)
#                 loss = self.multiple_negative_ranking_loss(pos_scores, neg_scores, target)
#             else:
#                 # scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

#                 # scores = scores.view(q_reps.size(0), -1)
#                 # target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
#                 # loss = self.compute_loss(scores, target)

#                 pos_scores = torch.matmul(q_reps, positive_reps.T)  # Similarity with positive example
#                 neg_scores = torch.matmul(q_reps, negative_reps.T)  # Similarity with negative examples
                
#                 pos_scores = pos_scores.view(q_reps.size(0), -1)
#                 neg_scores = neg_scores.view(q_reps.size(0), -1)
#                 target = torch.ones_like(pos_scores)
#                 loss = self.multiple_negative_ranking_loss(pos_scores, neg_scores, target)
#         else:
#             # During evaluation, just return similarity scores
#             scores = self.compute_similarity(q_reps, p_reps)
#             loss = None
        
#         return EncoderOutput(
#             loss=loss,
#             scores=scores,
#             q_reps=q_reps,
#             p_reps=p_reps,
#         )

#     # def compute_loss(self, scores, target):
#     #     return self.cross_entropy(scores, target)

   

#     def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
#         if t is None:
#             return None
#         t = t.contiguous()

#         all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
#         dist.all_gather(all_tensors, t)

#         all_tensors[self.process_rank] = t
#         all_tensors = torch.cat(all_tensors, dim=0)

#         return all_tensors

#     def save(self, output_dir: str):
#         state_dict = self.model.state_dict()
#         state_dict = type(state_dict)(
#             {k: v.clone().cpu()
#              for k,
#                  v in state_dict.items()})
#         self.model.save_pretrained(output_dir, state_dict=state_dict)

