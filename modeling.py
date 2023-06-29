import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import gelu, gelu_new
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
    LongformerPreTrainedModel,
    get_linear_schedule_with_warmup,
)


class IncrementalContextBert(BertPreTrainedModel):

    def __init__(self, config, num_embeddings, up_vocab):
        super().__init__(config)

        self.bert = BertModel(config)

        if config.do_shrink:
            self.embedding = nn.Embedding(num_embeddings, config.inner_size)
            self.to_hidden_size = nn.Linear(config.inner_size, config.hidden_size)
            self.to_inner_size = nn.Linear(config.hidden_size, config.inner_size)
        else:
            self.embedding = nn.Embedding(num_embeddings, config.hidden_size)

        self.multi_head_attention = torch.nn.MultiheadAttention(config.hidden_size, config.attention_heads)

        # Linear layers used to transform cls token, user and product embeddings
        self.linear_t = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_u = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_p = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.linear_update = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        self.linear_f = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.linear_g = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.gelu = gelu
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        # Classification layer
        self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels)

        # An empirical initializad number, still needed to be explored
        self.alpha = nn.Parameter(torch.tensor(-10, dtype=torch.float), requires_grad=True)

        self.up_vocab = up_vocab

        self.init_weights()

    def forward(self, inputs, user_product, up_indices=None, up_embeddings=None):
        if up_indices is not None and up_embeddings is not None:
            p_up_embeddings = self.embedding(up_indices)
            update_embeddings = p_up_embeddings + self.sigmoid(self.alpha)*up_embeddings
            with torch.no_grad():
                self.embedding.weight.index_copy(0, up_indices, update_embeddings)

        outputs = self.bert(**inputs)

        last_hidden_states, cls_hidden_states = outputs[0].transpose(0, 1), outputs[1]

        up_embeddings = self.embedding(user_product)

        if self.config.do_shrink:
            up_embeddings = self.to_hidden_size(up_embeddings)

        att_up = self.multi_head_attention(up_embeddings.transpose(0, 1), last_hidden_states, last_hidden_states)
        att_u, att_p = att_up[0][0, :, :], att_up[0][1, :, :]

        z_cls = self.sigmoid(self.linear_t(cls_hidden_states))
        z_att_u, z_att_p = self.sigmoid(self.linear_u(att_u)), self.sigmoid(self.linear_p(att_p))

        z_u = self.sigmoid(z_cls + z_att_u)
        z_p = self.sigmoid(z_cls + z_att_p)

        cls_input = cls_hidden_states + z_u * att_u + z_p * att_p

        logits = self.classifier(cls_input)
        # logits = self.softmax(logits)

        new_up_embeddings = torch.cat([z_att_u, z_att_p], dim=0)

        if self.config.do_shrink:
            new_up_embeddings = self.to_inner_size(new_up_embeddings)

        return logits, user_product.view(-1).detach(), new_up_embeddings


class CrossContextBert(BertPreTrainedModel):
    def __init__(self, config, u_num_embeddings, p_num_embeddings):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config) if 'longformer' not in config.model_type else LongformerModel(config)

        self.user_nums = u_num_embeddings
        self.prod_nums = p_num_embeddings

        self.user_embedding = nn.Embedding(u_num_embeddings, config.user_emb_size)
        self.product_embedding = nn.Embedding(p_num_embeddings, config.product_emb_size)
        self.user_textual_embedding = nn.Embedding(u_num_embeddings, config.user_emb_size)
        self.product_textual_embedding = nn.Embedding(p_num_embeddings, config.product_emb_size)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        is_batch_first = True
        self.u_multi_head_attention = nn.MultiheadAttention(config.hidden_size, config.attention_heads,
                                                            dropout=attn_dropout,
                                                            add_bias_kv=is_add_bias_kv,
                                                            add_zero_attn=is_add_zero_attn,
                                                            batch_first=is_batch_first)
        self.p_multi_head_attention = nn.MultiheadAttention(config.hidden_size, config.attention_heads,
                                                            dropout=attn_dropout,
                                                            add_bias_kv=is_add_bias_kv,
                                                            add_zero_attn=is_add_zero_attn,
                                                            batch_first=is_batch_first)

        self.uu_multi_head_attention = nn.MultiheadAttention(config.user_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.up_multi_head_attention = nn.MultiheadAttention(config.product_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.pu_multi_head_attention = nn.MultiheadAttention(config.user_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.pp_multi_head_attention = nn.MultiheadAttention(config.product_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)

        is_bias = True
        # self.transform_context = nn.Linear(in_features=5*config.hidden_size +
        #                                                2 * config.user_emb_size + 2 * config.product_emb_size,
        #                                    out_features=config.hidden_size, bias=is_bias)

        # self.combine_user = nn.Linear(in_features=config.user_emb_size + config.hidden_size,
        #                             out_features=config.user_emb_size, bias=is_bias)
        # self.combine_prod = nn.Linear(in_features=config.product_emb_size + config.hidden_size,
        #                             out_features=config.product_emb_size, bias=is_bias)

        self.fuse_u_cls = nn.Linear(in_features=config.user_emb_size + config.hidden_size,
                                    out_features=config.user_emb_size, bias=is_bias)
        self.fuse_p_cls = nn.Linear(in_features=config.product_emb_size + config.hidden_size,
                                    out_features=config.product_emb_size, bias=is_bias)
        self.transform_context = nn.Linear(in_features=5 * config.hidden_size,
                                           out_features=config.hidden_size, bias=is_bias)

        self.transform_u_to_hidden = nn.Linear(in_features=config.user_emb_size,
                                               out_features=config.hidden_size, bias=is_bias)
        self.transform_p_to_hidden = nn.Linear(in_features=config.product_emb_size,
                                               out_features=config.hidden_size, bias=is_bias)

        self.transform_u_to_p = nn.Linear(in_features=config.user_emb_size,
                                          out_features=config.product_emb_size, bias=is_bias)
        self.transform_p_to_u = nn.Linear(in_features=config.product_emb_size,
                                          out_features=config.user_emb_size, bias=is_bias)

        self.transform_u_to_cls = nn.Linear(in_features=config.user_emb_size,
                                            out_features=config.hidden_size)
        self.transform_p_to_cls = nn.Linear(in_features=config.product_emb_size,
                                            out_features=config.hidden_size)

        self.transform_to_classifier = nn.Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size)

        # self.transform_to_classifier = nn.Linear(in_features=4*config.hidden_size+
        #                                                      config.user_emb_size+config.product_emb_size,
        #                                          out_features=config.hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.gelu = gelu
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Classification layer
        self.classifier = nn.Linear(in_features=config.hidden_size,
                                    out_features=config.num_labels)

        self.coef = 0.1

        self.init_weights()

    def forward(self, inputs, user_ind, product_ind):
        # if up_indices is not None and up_embeddings is not None:
        #     p_up_embeddings = self.embedding(up_indices)
        #     update_embeddings = p_up_embeddings + self.sigmoid(self.alpha)*up_embeddings
        #     #with torch.no_grad():
        #     self.embedding.weight.index_copy(0, up_indices, update_embeddings)

        outputs = self.bert(**inputs)

        last_hidden_states, cls_hidden_states = outputs[0], outputs[1]

        u_embeddings = self.user_embedding(user_ind) \
            # + self.coef * self.user_textual_embedding(user_ind)
        p_embeddings = self.product_embedding(product_ind) \
            # + self.coef * self.product_textual_embedding(product_ind)

        '''inject textual embedding'''
        # u_embeddings = self.combine_user(torch.cat([u_embeddings,
        #                                             self.user_textual_embedding(user_ind)], dim=-1))
        # p_embeddings = self.combine_user(torch.cat([p_embeddings,
        #                                             self.product_textual_embedding(product_ind)], dim=-1))

        u_embeddings = self.fuse_u_cls(torch.cat([u_embeddings, cls_hidden_states.unsqueeze(1)], dim=-1))
        p_embeddings = self.fuse_p_cls(torch.cat([p_embeddings, cls_hidden_states.unsqueeze(1)], dim=-1))

        '''inject textual embedding'''
        # user_matrix = self.combine_user(torch.cat([self.user_embedding.weight,
        #                                            self.user_textual_embedding.weight], dim=-1))
        # prod_matrix = self.combine_user(torch.cat([self.product_embedding.weight,
        #                                            self.product_textual_embedding.weight], dim=-1))
        #
        # user_matrix = user_matrix.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)
        # prod_matrix = prod_matrix.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)

        user_matrix = self.user_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1) \
            # + self.coef * self.user_textual_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)
        prod_matrix = self.product_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1) \
            # + self.coef * self.product_textual_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1,
        #                                                                         1)

        user_masks = torch.zeros(u_embeddings.size(0) * self.config.attention_heads,
                                 1, self.user_nums)
        prod_masks = torch.zeros(p_embeddings.size(0) * self.config.attention_heads,
                                 1, self.prod_nums)

        user_masks = self.assign_attn_masks(user_ind, user_masks).bool().cuda()
        prod_masks = self.assign_attn_masks(product_ind, prod_masks).bool().cuda()

        attn_uu = self.uu_multi_head_attention(u_embeddings, user_matrix, user_matrix,
                                               attn_mask=user_masks)[0].squeeze(1)
        attn_up = self.up_multi_head_attention(self.transform_u_to_p(u_embeddings)
                                               , prod_matrix, prod_matrix)[0].squeeze(1)
        attn_pu = self.pu_multi_head_attention(self.transform_p_to_u(p_embeddings)
                                               , user_matrix, user_matrix)[0].squeeze(1)
        attn_pp = self.pp_multi_head_attention(p_embeddings, prod_matrix, prod_matrix,
                                               attn_mask=prod_masks)[0].squeeze(1)

        text_masks = (1 - inputs['attention_mask']).unsqueeze(1). \
            repeat(1, self.config.attention_heads, 1). \
            view(u_embeddings.size(0) * self.config.attention_heads, 1, -1).bool()
        text_masks[:, :, 0] = True

        u_transform, p_transform = self.transform_u_to_hidden(u_embeddings), \
                                   self.transform_p_to_hidden(p_embeddings)
        att_u = self.u_multi_head_attention(u_transform,
                                            last_hidden_states,
                                            last_hidden_states, attn_mask=text_masks)[0].squeeze(1)
        att_p = self.p_multi_head_attention(p_transform,
                                            last_hidden_states,
                                            last_hidden_states, attn_mask=text_masks)[0].squeeze(1)

        u_transform, p_transform = u_transform.squeeze(), p_transform.squeeze()

        # cls_features = torch.cat([cls_hidden_states, att_u, att_p, u_transform, p_transform,
        #                           attn_uu, attn_up, attn_pu, attn_pp], dim=-1)

        # # early injection of u_transform and p_transform
        # cls_features = torch.cat([cls_hidden_states + u_transform + p_transform, attn_uu, attn_up, attn_pu, attn_pp], dim=-1)
        #
        # cls_input = self.transform_context(cls_features) + att_u + att_p + cls_hidden_states

        # late injection of u_transform and p_transform
        cls_features = torch.cat([cls_hidden_states, attn_uu, attn_up, attn_pu, attn_pp], dim=-1)

        cls_input = self.transform_context(cls_features) + att_u + att_p + u_transform + p_transform + cls_hidden_states

        # without user-product coattention
        # cls_input = att_u + att_p + u_transform + p_transform + cls_hidden_states

        # without user-product coattention and distinguishing user and product embedding
        # cls_input = att_u + att_p + cls_hidden_states

        # with user and product information
        # cls_input = cls_hidden_states

        # cls_input = self.transform_context(cls_features) + att_u + att_p + \
        #             self.transform_u_to_cls(attn_uu + attn_pu) + self.transform_p_to_cls(attn_up + attn_pp) + \
        #             u_transform + p_transform + cls_hidden_states

        # cls_input = self.transform_to_classifier(cls_input)
        logits = self.classifier(cls_input)

        return logits, user_ind, product_ind, cls_hidden_states

    def assign_attn_masks(self, mask_ind, attn_mask_matrix):
        masks = list(mask_ind.detach().cpu().numpy())
        for i, mask in enumerate(masks):
            for j in range(self.config.attention_heads):
                attn_mask_matrix[i * self.config.attention_heads + j, :, mask] = 1

        return attn_mask_matrix


class CrossContextBertLen(BertPreTrainedModel):
    def __init__(self, config, u_num_embeddings, p_num_embeddings):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config) if 'longformer' not in config.model_type else LongformerModel(config)

        self.user_nums = u_num_embeddings
        self.prod_nums = p_num_embeddings

        self.user_embedding = nn.Embedding(u_num_embeddings, config.user_emb_size)
        self.product_embedding = nn.Embedding(p_num_embeddings, config.product_emb_size)
        self.user_textual_embedding = nn.Embedding(u_num_embeddings, config.user_emb_size)
        self.product_textual_embedding = nn.Embedding(p_num_embeddings, config.product_emb_size)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        is_batch_first = True
        self.u_multi_head_attention = nn.MultiheadAttention(config.hidden_size, config.attention_heads,
                                                            dropout=attn_dropout,
                                                            add_bias_kv=is_add_bias_kv,
                                                            add_zero_attn=is_add_zero_attn,
                                                            batch_first=is_batch_first)
        self.p_multi_head_attention = nn.MultiheadAttention(config.hidden_size, config.attention_heads,
                                                            dropout=attn_dropout,
                                                            add_bias_kv=is_add_bias_kv,
                                                            add_zero_attn=is_add_zero_attn,
                                                            batch_first=is_batch_first)

        self.uu_multi_head_attention = nn.MultiheadAttention(config.user_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.up_multi_head_attention = nn.MultiheadAttention(config.product_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.pu_multi_head_attention = nn.MultiheadAttention(config.user_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.pp_multi_head_attention = nn.MultiheadAttention(config.product_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)

        is_bias = True
        # self.transform_context = nn.Linear(in_features=5*config.hidden_size +
        #                                                2 * config.user_emb_size + 2 * config.product_emb_size,
        #                                    out_features=config.hidden_size, bias=is_bias)

        # self.combine_user = nn.Linear(in_features=config.user_emb_size + config.hidden_size,
        #                             out_features=config.user_emb_size, bias=is_bias)
        # self.combine_prod = nn.Linear(in_features=config.product_emb_size + config.hidden_size,
        #                             out_features=config.product_emb_size, bias=is_bias)

        self.fuse_u_cls = nn.Linear(in_features=config.user_emb_size + config.hidden_size,
                                    out_features=config.user_emb_size, bias=is_bias)
        self.fuse_p_cls = nn.Linear(in_features=config.product_emb_size + config.hidden_size,
                                    out_features=config.product_emb_size, bias=is_bias)
        self.transform_context = nn.Linear(in_features=5 * config.hidden_size,
                                           out_features=config.hidden_size, bias=is_bias)

        self.transform_u_to_hidden = nn.Linear(in_features=config.user_emb_size,
                                               out_features=config.hidden_size, bias=is_bias)
        self.transform_p_to_hidden = nn.Linear(in_features=config.product_emb_size,
                                               out_features=config.hidden_size, bias=is_bias)

        self.transform_u_to_p = nn.Linear(in_features=config.user_emb_size,
                                          out_features=config.product_emb_size, bias=is_bias)
        self.transform_p_to_u = nn.Linear(in_features=config.product_emb_size,
                                          out_features=config.user_emb_size, bias=is_bias)

        self.transform_u_to_cls = nn.Linear(in_features=config.user_emb_size,
                                            out_features=config.hidden_size)
        self.transform_p_to_cls = nn.Linear(in_features=config.product_emb_size,
                                            out_features=config.hidden_size)

        self.transform_to_classifier = nn.Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size)

        # self.transform_to_classifier = nn.Linear(in_features=4*config.hidden_size+
        #                                                      config.user_emb_size+config.product_emb_size,
        #                                          out_features=config.hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.gelu = gelu
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Classification layer
        self.classifier = nn.Linear(in_features=config.hidden_size,
                                    out_features=config.num_labels)

        self.coef = 0.1

        self.init_weights()

    def forward(self, inputs, user_ind, product_ind):
        # if up_indices is not None and up_embeddings is not None:
        #     p_up_embeddings = self.embedding(up_indices)
        #     update_embeddings = p_up_embeddings + self.sigmoid(self.alpha)*up_embeddings
        #     #with torch.no_grad():
        #     self.embedding.weight.index_copy(0, up_indices, update_embeddings)

        outputs = self.bert(**inputs)

        last_hidden_states, cls_hidden_states = outputs[0], outputs[1]

        u_embeddings = self.user_embedding(user_ind) \
            # + self.coef * self.user_textual_embedding(user_ind)
        p_embeddings = self.product_embedding(product_ind) \
            # + self.coef * self.product_textual_embedding(product_ind)

        '''inject textual embedding'''
        # u_embeddings = self.combine_user(torch.cat([u_embeddings,
        #                                             self.user_textual_embedding(user_ind)], dim=-1))
        # p_embeddings = self.combine_user(torch.cat([p_embeddings,
        #                                             self.product_textual_embedding(product_ind)], dim=-1))

        u_embeddings = self.fuse_u_cls(torch.cat([u_embeddings, cls_hidden_states.unsqueeze(1)], dim=-1))
        p_embeddings = self.fuse_p_cls(torch.cat([p_embeddings, cls_hidden_states.unsqueeze(1)], dim=-1))

        '''inject textual embedding'''
        # user_matrix = self.combine_user(torch.cat([self.user_embedding.weight,
        #                                            self.user_textual_embedding.weight], dim=-1))
        # prod_matrix = self.combine_user(torch.cat([self.product_embedding.weight,
        #                                            self.product_textual_embedding.weight], dim=-1))
        #
        # user_matrix = user_matrix.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)
        # prod_matrix = prod_matrix.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)

        user_matrix = self.user_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1) \
            # + self.coef * self.user_textual_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)
        prod_matrix = self.product_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1) \
            # + self.coef * self.product_textual_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1,
        #                                                                         1)

        user_masks = torch.zeros(u_embeddings.size(0) * self.config.attention_heads,
                                 1, self.user_nums)
        prod_masks = torch.zeros(p_embeddings.size(0) * self.config.attention_heads,
                                 1, self.prod_nums)

        user_masks = self.assign_attn_masks(user_ind, user_masks).bool().cuda()
        prod_masks = self.assign_attn_masks(product_ind, prod_masks).bool().cuda()

        attn_uu = self.uu_multi_head_attention(u_embeddings, user_matrix, user_matrix,
                                               attn_mask=user_masks)[0].squeeze(1)
        attn_up = self.up_multi_head_attention(self.transform_u_to_p(u_embeddings)
                                               , prod_matrix, prod_matrix)[0].squeeze(1)
        attn_pu = self.pu_multi_head_attention(self.transform_p_to_u(p_embeddings)
                                               , user_matrix, user_matrix)[0].squeeze(1)
        attn_pp = self.pp_multi_head_attention(p_embeddings, prod_matrix, prod_matrix,
                                               attn_mask=prod_masks)[0].squeeze(1)

        text_masks = (1 - inputs['attention_mask']).unsqueeze(1). \
            repeat(1, self.config.attention_heads, 1). \
            view(u_embeddings.size(0) * self.config.attention_heads, 1, -1).bool()
        text_masks[:, :, 0] = True

        u_transform, p_transform = self.transform_u_to_hidden(u_embeddings), \
                                   self.transform_p_to_hidden(p_embeddings)
        att_u = self.u_multi_head_attention(u_transform,
                                            last_hidden_states,
                                            last_hidden_states, attn_mask=text_masks)[0].squeeze(1)
        att_p = self.p_multi_head_attention(p_transform,
                                            last_hidden_states,
                                            last_hidden_states, attn_mask=text_masks)[0].squeeze(1)

        u_transform, p_transform = u_transform.squeeze(), p_transform.squeeze()

        # cls_features = torch.cat([cls_hidden_states, att_u, att_p, u_transform, p_transform,
        #                           attn_uu, attn_up, attn_pu, attn_pp], dim=-1)

        # # early injection of u_transform and p_transform
        # cls_features = torch.cat([cls_hidden_states + u_transform + p_transform, attn_uu, attn_up, attn_pu, attn_pp], dim=-1)
        #
        # cls_input = self.transform_context(cls_features) + att_u + att_p + cls_hidden_states

        # late injection of u_transform and p_transform
        cls_features = torch.cat([cls_hidden_states, attn_uu, attn_up, attn_pu, attn_pp], dim=-1)

        cls_input = self.transform_context(cls_features) + att_u + att_p + u_transform + p_transform + cls_hidden_states

        # without user-product coattention
        # cls_input = att_u + att_p + u_transform + p_transform + cls_hidden_states

        # without user-product coattention and distinguishing user and product embedding
        # cls_input = att_u + att_p + cls_hidden_states

        # with user and product information
        # cls_input = cls_hidden_states

        # cls_input = self.transform_context(cls_features) + att_u + att_p + \
        #             self.transform_u_to_cls(attn_uu + attn_pu) + self.transform_p_to_cls(attn_up + attn_pp) + \
        #             u_transform + p_transform + cls_hidden_states

        # cls_input = self.transform_to_classifier(cls_input)
        logits = self.classifier(cls_input)

        return logits, user_ind, product_ind, cls_hidden_states

    def assign_attn_masks(self, mask_ind, attn_mask_matrix):
        masks = list(mask_ind.detach().cpu().numpy())
        for i, mask in enumerate(masks):
            for j in range(self.config.attention_heads):
                attn_mask_matrix[i * self.config.attention_heads + j, :, mask] = 1

        return attn_mask_matrix


class CrossContextBertRatio(BertPreTrainedModel):
    def __init__(self, config, u_num_embeddings, p_num_embeddings):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config) if 'longformer' not in config.model_type else LongformerModel(config)

        self.user_nums = u_num_embeddings
        self.prod_nums = p_num_embeddings

        self.user_embedding = nn.Embedding(u_num_embeddings, config.user_emb_size)
        self.product_embedding = nn.Embedding(p_num_embeddings, config.product_emb_size)
        self.user_textual_embedding = nn.Embedding(u_num_embeddings, config.user_emb_size)
        self.product_textual_embedding = nn.Embedding(p_num_embeddings, config.product_emb_size)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        is_batch_first = True
        self.u_multi_head_attention = nn.MultiheadAttention(config.hidden_size, config.attention_heads,
                                                            dropout=attn_dropout,
                                                            add_bias_kv=is_add_bias_kv,
                                                            add_zero_attn=is_add_zero_attn,
                                                            batch_first=is_batch_first)
        self.p_multi_head_attention = nn.MultiheadAttention(config.hidden_size, config.attention_heads,
                                                            dropout=attn_dropout,
                                                            add_bias_kv=is_add_bias_kv,
                                                            add_zero_attn=is_add_zero_attn,
                                                            batch_first=is_batch_first)

        self.uu_multi_head_attention = nn.MultiheadAttention(config.user_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.up_multi_head_attention = nn.MultiheadAttention(config.product_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.pu_multi_head_attention = nn.MultiheadAttention(config.user_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)
        self.pp_multi_head_attention = nn.MultiheadAttention(config.product_emb_size, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn,
                                                             batch_first=is_batch_first)

        is_bias = True
        # self.transform_context = nn.Linear(in_features=5*config.hidden_size +
        #                                                2 * config.user_emb_size + 2 * config.product_emb_size,
        #                                    out_features=config.hidden_size, bias=is_bias)

        # self.combine_user = nn.Linear(in_features=config.user_emb_size + config.hidden_size,
        #                             out_features=config.user_emb_size, bias=is_bias)
        # self.combine_prod = nn.Linear(in_features=config.product_emb_size + config.hidden_size,
        #                             out_features=config.product_emb_size, bias=is_bias)

        self.fuse_u_cls = nn.Linear(in_features=config.user_emb_size + config.hidden_size,
                                    out_features=config.user_emb_size, bias=is_bias)
        self.fuse_p_cls = nn.Linear(in_features=config.product_emb_size + config.hidden_size,
                                    out_features=config.product_emb_size, bias=is_bias)
        self.transform_context = nn.Linear(in_features=5 * config.hidden_size,
                                           out_features=config.hidden_size, bias=is_bias)

        self.transform_u_to_hidden = nn.Linear(in_features=config.user_emb_size,
                                               out_features=config.hidden_size, bias=is_bias)
        self.transform_p_to_hidden = nn.Linear(in_features=config.product_emb_size,
                                               out_features=config.hidden_size, bias=is_bias)

        self.transform_u_to_p = nn.Linear(in_features=config.user_emb_size,
                                          out_features=config.product_emb_size, bias=is_bias)
        self.transform_p_to_u = nn.Linear(in_features=config.product_emb_size,
                                          out_features=config.user_emb_size, bias=is_bias)

        self.transform_u_to_cls = nn.Linear(in_features=config.user_emb_size,
                                            out_features=config.hidden_size)
        self.transform_p_to_cls = nn.Linear(in_features=config.product_emb_size,
                                            out_features=config.hidden_size)

        self.transform_to_classifier = nn.Linear(in_features=config.hidden_size,
                                                 out_features=config.hidden_size)

        # self.transform_to_classifier = nn.Linear(in_features=4*config.hidden_size+
        #                                                      config.user_emb_size+config.product_emb_size,
        #                                          out_features=config.hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.gelu = gelu
        self.relu = nn.ReLU()
        self.celu = nn.CELU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Classification layer
        self.classifier = nn.Linear(in_features=config.hidden_size,
                                    out_features=config.num_labels)

        self.coef = 0.1

        self.init_weights()

    def forward(self, inputs, user_ind, product_ind):
        # if up_indices is not None and up_embeddings is not None:
        #     p_up_embeddings = self.embedding(up_indices)
        #     update_embeddings = p_up_embeddings + self.sigmoid(self.alpha)*up_embeddings
        #     #with torch.no_grad():
        #     self.embedding.weight.index_copy(0, up_indices, update_embeddings)

        outputs = self.bert(**inputs)

        last_hidden_states, cls_hidden_states = outputs[0], outputs[1]

        u_embeddings = self.user_embedding(user_ind) \
            # + self.coef * self.user_textual_embedding(user_ind)
        p_embeddings = self.product_embedding(product_ind) \
            # + self.coef * self.product_textual_embedding(product_ind)

        '''inject textual embedding'''
        # u_embeddings = self.combine_user(torch.cat([u_embeddings,
        #                                             self.user_textual_embedding(user_ind)], dim=-1))
        # p_embeddings = self.combine_user(torch.cat([p_embeddings,
        #                                             self.product_textual_embedding(product_ind)], dim=-1))

        u_embeddings = self.fuse_u_cls(torch.cat([u_embeddings, cls_hidden_states.unsqueeze(1)], dim=-1))
        p_embeddings = self.fuse_p_cls(torch.cat([p_embeddings, cls_hidden_states.unsqueeze(1)], dim=-1))

        '''inject textual embedding'''
        # user_matrix = self.combine_user(torch.cat([self.user_embedding.weight,
        #                                            self.user_textual_embedding.weight], dim=-1))
        # prod_matrix = self.combine_user(torch.cat([self.product_embedding.weight,
        #                                            self.product_textual_embedding.weight], dim=-1))
        #
        # user_matrix = user_matrix.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)
        # prod_matrix = prod_matrix.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)

        user_matrix = self.user_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1) \
            # + self.coef * self.user_textual_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1)
        prod_matrix = self.product_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1, 1) \
            # + self.coef * self.product_textual_embedding.weight.unsqueeze(0).repeat(u_embeddings.size(0), 1,
        #                                                                         1)

        user_masks = torch.zeros(u_embeddings.size(0) * self.config.attention_heads,
                                 1, self.user_nums)
        prod_masks = torch.zeros(p_embeddings.size(0) * self.config.attention_heads,
                                 1, self.prod_nums)

        user_masks = self.assign_attn_masks(user_ind, user_masks).bool().cuda()
        prod_masks = self.assign_attn_masks(product_ind, prod_masks).bool().cuda()

        attn_uu = self.uu_multi_head_attention(u_embeddings, user_matrix, user_matrix,
                                               attn_mask=user_masks)[0].squeeze(1)
        attn_up = self.up_multi_head_attention(self.transform_u_to_p(u_embeddings)
                                               , prod_matrix, prod_matrix)[0].squeeze(1)
        attn_pu = self.pu_multi_head_attention(self.transform_p_to_u(p_embeddings)
                                               , user_matrix, user_matrix)[0].squeeze(1)
        attn_pp = self.pp_multi_head_attention(p_embeddings, prod_matrix, prod_matrix,
                                               attn_mask=prod_masks)[0].squeeze(1)

        text_masks = (1 - inputs['attention_mask']).unsqueeze(1). \
            repeat(1, self.config.attention_heads, 1). \
            view(u_embeddings.size(0) * self.config.attention_heads, 1, -1).bool()
        text_masks[:, :, 0] = True

        u_transform, p_transform = self.transform_u_to_hidden(u_embeddings), \
                                   self.transform_p_to_hidden(p_embeddings)
        att_u = self.u_multi_head_attention(u_transform,
                                            last_hidden_states,
                                            last_hidden_states, attn_mask=text_masks)[0].squeeze(1)
        att_p = self.p_multi_head_attention(p_transform,
                                            last_hidden_states,
                                            last_hidden_states, attn_mask=text_masks)[0].squeeze(1)

        u_transform, p_transform = u_transform.squeeze(), p_transform.squeeze()

        # cls_features = torch.cat([cls_hidden_states, att_u, att_p, u_transform, p_transform,
        #                           attn_uu, attn_up, attn_pu, attn_pp], dim=-1)

        # # early injection of u_transform and p_transform
        # cls_features = torch.cat([cls_hidden_states + u_transform + p_transform, attn_uu, attn_up, attn_pu, attn_pp], dim=-1)
        #
        # cls_input = self.transform_context(cls_features) + att_u + att_p + cls_hidden_states

        # late injection of u_transform and p_transform
        cls_features = torch.cat([cls_hidden_states, attn_uu, attn_up, attn_pu, attn_pp], dim=-1)

        cls_input = self.transform_context(cls_features) + att_u + att_p + u_transform + p_transform + cls_hidden_states

        # without user-product coattention
        # cls_input = att_u + att_p + u_transform + p_transform + cls_hidden_states

        # without user-product coattention and distinguishing user and product embedding
        # cls_input = att_u + att_p + cls_hidden_states

        # cls_input = self.transform_context(cls_features) + att_u + att_p + \
        #             self.transform_u_to_cls(attn_uu + attn_pu) + self.transform_p_to_cls(attn_up + attn_pp) + \
        #             u_transform + p_transform + cls_hidden_states

        # cls_input = self.transform_to_classifier(cls_input)
        logits = self.classifier(cls_input)

        return logits, user_ind, product_ind, cls_hidden_states

    def assign_attn_masks(self, mask_ind, attn_mask_matrix):
        masks = list(mask_ind.detach().cpu().numpy())
        for i, mask in enumerate(masks):
            for j in range(self.config.attention_heads):
                attn_mask_matrix[i * self.config.attention_heads + j, :, mask] = 1

        return attn_mask_matrix


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
