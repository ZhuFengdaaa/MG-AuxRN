
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args

class ProgressIndicator(nn.Module):
    def __init__(self):
        super(ProgressIndicator, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class MatchingNetwork(nn.Module):
    def __init__(self):
        super(MatchingNetwork, self).__init__()
        hidden_size = args.rnn_dim
        if args.mat_mul:
            self.fc1 = nn.Linear(hidden_size, hidden_size)
        else:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h1, ctx):
        if args.mat_norm:
            h1 = h1 / (torch.norm(h1, dim=1).unsqueeze(1)+1e-6)
            ctx = ctx / (torch.norm(ctx, dim=1).unsqueeze(1)+1e-6)
        if args.mat_mul:
            h = h1 * ctx
        else:
            h = torch.cat((h1, ctx), dim=1)
        h = self.relu1(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        # h = torch.mean(h, dim=1) # pooling, harm performance
        return h

class FeaturePredictor(nn.Module):
    def __init__(self):
        super(FeaturePredictor, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, args.feature_size)

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.fc2(h)
        return h

class AnglePredictor(nn.Module):
    def __init__(self):
        super(AnglePredictor, self).__init__()
        hidden_size = args.rnn_dim
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h = self.relu1(self.fc1(h))
        h = self.fc2(h)
        return h

class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx, 
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, 
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention. 

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn

class MultiHeadSelfAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self,num_heads, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.linear_in = []
        for i in range(self.num_heads):
            self.linear_in.append(nn.Linear(query_dim,ctx_dim,bias=False).cuda())
        self.sm = nn.Softmax()
        self.linear_concat_out = nn.Linear(self.num_heads*ctx_dim+query_dim, query_dim, bias=False)
        # self.linear_out = nn.Linear(self.num_heads*ctx_dim, ctx_dim,bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        append_logit = []
        append_weighted_context = []
        append_attn = []

        for i in range(self.num_heads):

            target = self.linear_in[i](h).unsqueeze(2)  # batch x dim x 1

            # Get attention
            attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
            logit = attn

            if mask is not None:
                # -Inf masking prior to the softmax
                attn.masked_fill_(mask.bool(), -float('inf'))
            attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
            attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

            weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
            append_logit.append(logit)                        # num_head x batch x seq_len
            append_weighted_context.append(weighted_context)
            append_attn.append(attn)

        output_logit = torch.stack(append_logit)
        output_weighted_context = torch.cat(append_weighted_context,1)
        output_attn = torch.stack(append_attn)

        output_logit = output_logit.mean(dim=0)
        output_attn = output_attn.mean(dim=0)

        if not output_prob:
            output_attn = output_logit
        if output_tilde:
            h_tilde = torch.cat((output_weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_concat_out(h_tilde))
            return h_tilde, output_attn
        else:
            # output_weighted_context = self.linear_out(output_weighted_context)
            return output_weighted_context, output_attn

class Gate(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(Gate, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sg = nn.Sigmoid()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask.bool(), -float('inf'))
        if args.objInputMode == 'sg':
            attn = self.sg(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        elif args.objInputMode == 'tanh':
            attn = self.tanh(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.obj_feat_size = 0
        self.s_obj_feat_size = 0
        self.d_obj_feat_size = 0
        if args.sparseObj :
            print("Train in sparseObj cat %s, %s mode" % (args.catfeat, args.objInputMode))
            if args.catfeat == 'none':
                self.s_obj_angle_num = 0
            elif args.denseObj:
                self.s_obj_angle_num = 0
            else:
                self.s_obj_angle_num = 1
            self.s_obj_feat_size = args.instEmb + args.instHE * self.s_obj_angle_num
        if args.denseObj:
            print("Train in denseObj cat %s, %s mode" % (args.catfeat,args.objInputMode))
            if args.catfeat == 'none':
                self.d_obj_angle_num = 0
            else:
                self.d_obj_angle_num= 1
            self.d_obj_feat_size = args.feature_size+args.angle_feat_size*self.d_obj_angle_num
        self.obj_feat_size = self.s_obj_feat_size+self.d_obj_feat_size
        print('Obj feature size: %d + %d'%(self.s_obj_feat_size, self.d_obj_feat_size))
        if not args.denseObj and not args.sparseObj:
            print("Train in RN mode")
        self.view_feat_size = args.feature_size+args.angle_feat_size
        self.lstm_feature_size = self.view_feat_size + self.obj_feat_size
            # feature_size = args.feature_size+args.angle_feat_size
        print('LSTM feature size:%d + %d '%(self.view_feat_size, self.obj_feat_size))
        self.embedding_size = embedding_size
        # self.lstm_feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)

        if args.multiMode == "vis" and args.headNum > 1:
            self.lstm_feature_size = (self.view_feat_size)*args.headNum+self.obj_feat_size
            print("LSTM feature size: %d x %d + %d"%(self.view_feat_size, args.headNum, self.obj_feat_size))
            self.feat_att_layer = MultiHeadSelfAttention(args.headNum, hidden_size, args.feature_size + args.angle_feat_size)
            self.lstm = nn.LSTMCell(embedding_size+self.lstm_feature_size, hidden_size)
        else:
            self.lstm = nn.LSTMCell(embedding_size + self.lstm_feature_size, hidden_size)
            self.feat_att_layer = SoftDotAttention(hidden_size, args.feature_size + args.angle_feat_size)

        if args.denseObj:
            if args.objInputMode == 'sg' or 'tanh':
                self.dense_input_layer = Gate(hidden_size, self.d_obj_feat_size)
            elif args.objInputMode == 'sm':
                self.dense_input_layer = SoftDotAttention(hidden_size, self.d_obj_feat_size)
        if args.sparseObj:
            if args.objInputMode == 'sg' or 'tanh':
                self.sparse_input_layer = Gate(hidden_size, self.s_obj_feat_size)
            elif args.objInputMode == 'sm':
                self.sparse_input_layer = SoftDotAttention(hidden_size, self.s_obj_feat_size)
        if args.multiMode == 'can' and args.headNum > 1:
            self.candidate_att_layer = MultiHeadSelfAttention(args.headNum, hidden_size, args.feature_size+args.angle_feat_size)
        else:
            self.candidate_att_layer = SoftDotAttention(hidden_size, args.feature_size+args.angle_feat_size)
        # self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

        if args.multiMode == 'ins' and args.headNum > 1:
            self.attention_layer = MultiHeadSelfAttention(args.headNum,hidden_size, hidden_size)
        else:
            self.attention_layer = SoftDotAttention(hidden_size, hidden_size)

    def forward(self, action, cand_feat,
                prev_h1, c_0,
                ctx, ctx_mask=None,feature=None, sparseObj=None,denseObj=None,ObjFeature_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)
            if sparseObj is not None:
                if args.catfeat == 'none':
                    sparseObj = self.drop_env(sparseObj)
                else:
                    sparseObj[..., -args.instHE] = self.drop_env(sparseObj[..., -args.instHE])
            if denseObj is not None:
                if args.catfeat == 'none':
                    denseObj = self.drop_env(denseObj)
                else:
                    denseObj[..., -args.angle_feat_size] = self.drop_env(denseObj[...,-args.angle_feat_size])

        prev_h1_drop = self.drop(prev_h1)

        if args.sparseObj:
            sparse_input_feat, _ = self.sparse_input_layer(prev_h1_drop, sparseObj, mask=ObjFeature_mask,
                                                       output_tilde=False)
        if args.denseObj:
            dense_input_feat, _ = self.dense_input_layer(prev_h1_drop, denseObj, mask=ObjFeature_mask,
                                                         output_tilde=False)
        if args.sparseObj and (not args.denseObj):
            obj_input_feat = sparse_input_feat
        elif args.denseObj and (not args.sparseObj):
            obj_input_feat = dense_input_feat
        elif args.denseObj and args.sparseObj:
            obj_input_feat = torch.cat([sparse_input_feat, dense_input_feat],1)

        RN_attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        if args.sparseObj or args.denseObj:
            attn_feat = torch.cat([RN_attn_feat,obj_input_feat],1)
        else:
            attn_feat = RN_attn_feat

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()

class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x

class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


class SpeakerDecoder_SameLSTM(SpeakerDecoder):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super(SpeakerDecoder_SameLSTM, self).__init__(vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio)

    def forward(self, words, ctx, ctx_mask, ctx_w):
        # embeds = self.embedding(words)
        # embeds = self.drop(embeds)
        # x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = ctx_w
        # Get the size
        batchXlength = ctx_w.size(0) * ctx_w.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search
        x = self.drop(x)
        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(ctx_w.size(0), ctx_w.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit
