# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn
import random
import numpy as np
from modules.encoder import EncoderCNN, EncoderLabels
from modules.transformer_decoder import DecoderTransformer
from modules.multihead_attention import MultiheadAttention
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import pickle
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label2onehot(labels, pad_value):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    one_hot = torch.FloatTensor(labels.size(0), labels.size(1), pad_value + 1).zero_().to(device)
    one_hot.scatter_(2, inp_, 1)
    one_hot, _ = one_hot.max(dim=1)
    # remove pad position
    one_hot = one_hot[:, :-1]
    # eos position is always 0
    one_hot[:, 0] = 0

    return one_hot


def mask_from_eos(ids, eos_value, mult_before=True):
    mask = torch.ones(ids.size()).to(device).byte()
    mask_aux = torch.ones(ids.size(0)).to(device).byte()

    # find eos in ingredient prediction
    for idx in range(ids.size(1)):
        # force mask to have 1s in the first position to avoid division by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (ids[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask


def get_model(args, ingr_vocab_size, tool_vocab_size, action_vocab_size, instrs_vocab_size):

    # build ingredients, tools, actions embedding
    encoder_ingrs = EncoderLabels(args.embed_size, ingr_vocab_size,
                                  args.dropout_encoder, scale_grad=False).to(device)
    encoder_tools = EncoderLabels(args.embed_size, tool_vocab_size,
                                  args.dropout_encoder, scale_grad=False).to(device)
    encoder_actions = EncoderLabels(args.embed_size, action_vocab_size,
                                    args.dropout_encoder, scale_grad=False).to(device)

    # build image model
    encoder_image = EncoderCNN(args.embed_size, args.dropout_encoder, args.image_model)

    decoder = DecoderTransformer(args.embed_size, instrs_vocab_size,
                                 dropout=args.dropout_decoder_r, seq_length=args.maxseqlen,
                                 num_instrs=args.maxnuminstrs,
                                 attention_nheads=args.n_att, num_layers=args.transf_layers,
                                 normalize_before=True,
                                 normalize_inputs=False,
                                 last_ln=False,
                                 scale_embed_grad=False)

    ingr_decoder = DecoderTransformer(args.embed_size, ingr_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumlabels,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)

    tool_decoder = DecoderTransformer(args.embed_size, tool_vocab_size, dropout=args.dropout_decoder_i,
                                      seq_length=args.maxnumlabels,
                                      num_instrs=1, attention_nheads=args.n_att_ingrs,
                                      pos_embeddings=False,
                                      num_layers=args.transf_layers_ingrs,
                                      learned=False,
                                      normalize_before=True,
                                      normalize_inputs=True,
                                      last_ln=True,
                                      scale_embed_grad=False)

    action_decoder = DecoderTransformer(args.embed_size, action_vocab_size, dropout=args.dropout_decoder_i,
                                        seq_length=args.maxnumlabels,
                                        num_instrs=1, attention_nheads=args.n_att_ingrs,
                                        pos_embeddings=False,
                                        num_layers=args.transf_layers_ingrs,
                                        learned=False,
                                        normalize_before=True,
                                        normalize_inputs=True,
                                        last_ln=True,
                                        scale_embed_grad=False)

    # recipe loss
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size-1], reduce=False)

    # entity loss
    label_loss_ingr = nn.BCELoss(reduce=False)
    eos_loss_ingr = nn.BCELoss(reduce=False)

    label_loss_tool = nn.BCELoss(reduce=False)
    eos_loss_tool = nn.BCELoss(reduce=False)

    label_loss_action = nn.BCELoss(reduce=False)
    eos_loss_action = nn.BCELoss(reduce=False)

    model = InverseCookingModel(encoder_ingrs, encoder_tools, encoder_actions,
                                decoder, encoder_image,
                                ingr_decoder, tool_decoder, action_decoder,
                                crit=criterion,
                                crit_ingr=label_loss_ingr, crit_tool=label_loss_tool, crit_action=label_loss_action,
                                crit_eos_ingr=eos_loss_ingr, crit_eos_tool=eos_loss_tool, crit_eos_action=eos_loss_action,
                                pad_value=ingr_vocab_size-1, ingrs_only=args.ingrs_only,
                                recipe_only=args.recipe_only, label_smoothing=args.label_smoothing_ingr)

    return model


class InverseCookingModel(nn.Module):
    def __init__(self, ingredient_encoder, tool_encoder, action_encoder,
                 recipe_decoder, image_encoder,
                 ingr_decoder, tool_decoder, action_decoder,
                 crit=None,
                 crit_ingr=None, crit_tool=None, crit_action=None,
                 crit_eos_ingr=None, crit_eos_tool=None, crit_eos_action=None,
                 pad_value=0, ingrs_only=True,
                 recipe_only=False, label_smoothing=0.0):

        super(InverseCookingModel, self).__init__()

        # entity encoders
        self.ingredient_encoder = ingredient_encoder
        self.tool_encoder = tool_encoder
        self.action_encoder = action_encoder

        self.recipe_decoder = recipe_decoder
        self.image_encoder = image_encoder

        # entity decoders
        self.ingredient_decoder = ingr_decoder
        self.tool_decoder = tool_decoder
        self.action_decoder = action_decoder

        self.crit = crit

        # entity criteria
        self.crit_ingr = crit_ingr
        self.crit_tool = crit_tool
        self.crit_action = crit_action

        self.pad_value = pad_value
        self.ingrs_only = ingrs_only
        self.recipe_only = recipe_only

        # entity eos criteria
        self.crit_eos_ingr = crit_eos_ingr
        self.crit_eos_tool = crit_eos_tool
        self.crit_eos_action = crit_eos_action

        self.label_smoothing = label_smoothing

    def _entity_forward(self, entity_name, losses, img_features, entity_decoder, target_entities, crit_entity, crit_entity_eos):
        target_one_hot = label2onehot(target_entities, self.pad_value)
        target_one_hot_smooth = label2onehot(target_entities, self.pad_value)

        target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
        target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / target_one_hot_smooth.size(-1)

        # decode entities with transformer
        # autoregressive mode for entity decoder
        entity_ids, entity_logits = entity_decoder.sample(None, None, greedy=True,
                                                        temperature=1.0, img_features=img_features,
                                                        first_token_value=0, replacement=False)

        entity_logits = torch.nn.functional.softmax(entity_logits, dim=-1)

        # find idxs for eos entity
        # eos probability is the one assigned to the first position of the softmax
        eos = entity_logits[:, :, 0]
        target_eos = ((target_entities == 0) ^ (target_entities == self.pad_value))

        eos_pos = (target_entities == 0)
        eos_head = ((target_entities != self.pad_value) & (target_entities != 0))

        # select transformer steps to pool from
        mask_perminv = mask_from_eos(target_entities, eos_value=0, mult_before=False)
        entity_probs = entity_logits * mask_perminv.float().unsqueeze(-1)

        entity_probs, _ = torch.max(entity_probs, dim=1)

        # ignore predicted entity after eos in ground truth
        entity_ids[mask_perminv == 0] = self.pad_value

        entity_loss = crit_entity(entity_probs, target_one_hot_smooth)
        entity_loss = torch.mean(entity_loss, dim=-1)

        losses[f'{entity_name}_loss'] = entity_loss

        # cardinality penalty
        losses[f'{entity_name}_card_penalty'] = torch.abs((entity_probs*target_one_hot).sum(1) - target_one_hot.sum(1)) + \
                                    torch.abs((entity_probs*(1-target_one_hot)).sum(1))

        eos_loss = crit_entity_eos(eos, target_eos.float())

        mult = 1/2
        # eos loss is only computed for timesteps <= t_eos and equally penalizes 0s and 1s
        losses[f'{entity_name}_eos_loss'] = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
                                mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
        # iou
        pred_one_hot = label2onehot(entity_ids, self.pad_value)
        # iou sample during training is computed using the true eos position
        losses[f'{entity_name}_iou'] = softIoU(pred_one_hot, target_one_hot)

        return losses

    def _encode_entities(self, entity_encoder, target_entities):
        # encode gt entities
        target_entity_feats = entity_encoder(target_entities)

        target_entity_mask = mask_from_eos(target_entities, eos_value=0, mult_before=False)
        target_entity_mask = target_entity_mask.float().unsqueeze(1)

        return target_entity_feats, target_entity_mask

    def forward(self, img_inputs, captions,
                target_ingrs, target_tools, target_actions,
                sample=False, keep_cnn_gradients=False):

        if sample:
            return self.sample(img_inputs, greedy=True)

        targets = captions[:, 1:]
        targets = targets.contiguous().view(-1)

        img_features = self.image_encoder(img_inputs, keep_cnn_gradients)

        losses = {}

        ### TODO: REPLACE ######## 
        target_one_hot = label2onehot(target_ingrs, self.pad_value)
        target_one_hot_smooth = label2onehot(target_ingrs, self.pad_value)

        # ingredient prediction
        if not self.recipe_only:
            target_one_hot_smooth[target_one_hot_smooth == 1] = (1-self.label_smoothing)
            target_one_hot_smooth[target_one_hot_smooth == 0] = self.label_smoothing / target_one_hot_smooth.size(-1)

            # decode ingredients with transformer
            # autoregressive mode for ingredient decoder
            ingr_ids, ingr_logits = self.ingredient_decoder.sample(None, None, greedy=True,
                                                                   temperature=1.0, img_features=img_features,
                                                                   first_token_value=0, replacement=False)

            ingr_logits = torch.nn.functional.softmax(ingr_logits, dim=-1)

            # find idxs for eos ingredient
            # eos probability is the one assigned to the first position of the softmax
            eos = ingr_logits[:, :, 0]
            target_eos = ((target_ingrs == 0) ^ (target_ingrs == self.pad_value))

            eos_pos = (target_ingrs == 0)
            eos_head = ((target_ingrs != self.pad_value) & (target_ingrs != 0))

            # select transformer steps to pool from
            mask_perminv = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)
            ingr_probs = ingr_logits * mask_perminv.float().unsqueeze(-1)

            ingr_probs, _ = torch.max(ingr_probs, dim=1)

            # ignore predicted ingredients after eos in ground truth
            ingr_ids[mask_perminv == 0] = self.pad_value

            ingr_loss = self.crit_ingr(ingr_probs, target_one_hot_smooth)
            ingr_loss = torch.mean(ingr_loss, dim=-1)

            losses['ingr_loss'] = ingr_loss

            # cardinality penalty
            losses['card_penalty'] = torch.abs((ingr_probs*target_one_hot).sum(1) - target_one_hot.sum(1)) + \
                                     torch.abs((ingr_probs*(1-target_one_hot)).sum(1))

            eos_loss = self.crit_eos(eos, target_eos.float())

            mult = 1/2
            # eos loss is only computed for timesteps <= t_eos and equally penalizes 0s and 1s
            losses['eos_loss'] = mult*(eos_loss * eos_pos.float()).sum(1) / (eos_pos.float().sum(1) + 1e-6) + \
                                 mult*(eos_loss * eos_head.float()).sum(1) / (eos_head.float().sum(1) + 1e-6)
            # iou
            pred_one_hot = label2onehot(ingr_ids, self.pad_value)
            # iou sample during training is computed using the true eos position
            losses['iou'] = softIoU(pred_one_hot, target_one_hot)
        ### END: REPLACE ########

        losses = self._entity_forward('ingredients', losses, img_features, self.ingredient_decoder,
                                      target_ingrs, self.crit_ingr, self.crit_eos_ingr)
        losses = self._entity_forward('tools', losses, img_features, self.tool_decoder,
                                      target_tools, self.crit_tool, self.crit_eos_tool)
        losses = self._entity_forward('actions', losses, img_features, self.action_decoder,
                                      target_actions, self.crit_action, self.crit_eos_action)

        if self.ingrs_only:
            return losses

        # encode ingredients

        # TODO: REPLACE ###########
        target_ingr_feats = self.ingredient_encoder(target_ingrs)
        target_ingr_mask = mask_from_eos(target_ingrs, eos_value=0, mult_before=False)

        target_ingr_mask = target_ingr_mask.float().unsqueeze(1)
        ###### END REPLACE ########

        target_ingr_feats, target_ingr_mask = self._encode_entities(self.ingredient_encoder, target_ingrs)
        target_tool_feats, target_tool_mask = self._encode_entities(self.tool_encoder, target_tools)
        target_action_feats, target_action_mask = self._encode_entities(self.action_encoder, target_actions)

        outputs, ids = self.recipe_decoder(target_ingr_feats, target_ingr_mask, 
                                           target_tool_feats, target_tool_mask,
                                           target_action_feats, target_action_mask,
                                           captions, img_features)

        outputs = outputs[:, :-1, :].contiguous()
        outputs = outputs.view(outputs.size(0) * outputs.size(1), -1)

        loss = self.crit(outputs, targets)

        losses['recipe_loss'] = loss

        return losses

    def sample(self, img_inputs, greedy=True, temperature=1.0, beam=-1, true_ingrs=None):

        outputs = dict()

        img_features = self.image_encoder(img_inputs)

        if not self.recipe_only:
            ingr_ids, ingr_probs = self.ingredient_decoder.sample(None, None, greedy=True, temperature=temperature,
                                                                  beam=-1,
                                                                  img_features=img_features, first_token_value=0,
                                                                  replacement=False)

            # mask ingredients after finding eos
            sample_mask = mask_from_eos(ingr_ids, eos_value=0, mult_before=False)
            ingr_ids[sample_mask == 0] = self.pad_value

            outputs['ingr_ids'] = ingr_ids
            outputs['ingr_probs'] = ingr_probs.data

            mask = sample_mask
            input_mask = mask.float().unsqueeze(1)
            input_feats = self.ingredient_encoder(ingr_ids)

        if self.ingrs_only:
            return outputs

        # option during sampling to use the real ingredients and not the predicted ones to infer the recipe
        if true_ingrs is not None:
            input_mask = mask_from_eos(true_ingrs, eos_value=0, mult_before=False)
            true_ingrs[input_mask == 0] = self.pad_value
            input_feats = self.ingredient_encoder(true_ingrs)
            input_mask = input_mask.unsqueeze(1)

        ids, probs = self.recipe_decoder.sample(input_feats, input_mask, greedy, temperature, beam, img_features, 0,
                                                last_token_value=1)

        outputs['recipe_probs'] = probs.data
        outputs['recipe_ids'] = ids

        return outputs
