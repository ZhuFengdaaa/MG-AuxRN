
import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env import R2RBatch
from utils import padding_idx, add_idx, Tokenizer, get_sync_dir
import utils
import model
import param
from param import args
from collections import defaultdict


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        self.encoder = model.EncoderLSTM(tok.vocab_size(), args.wemb, enc_hidden_size, padding_idx,
                                         args.dropout, bidirectional=args.bidir).cuda()
        self.decoder = model.AttnDecoderLSTM(args.aemb, args.rnn_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.critic = model.Critic().cuda()
        self.models = (self.encoder, self.decoder, self.critic)

        # Optimizers
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer)

        if args.aux_option:
            if args.modspe:
                self.speaker_decoder = model.SpeakerDecoder_SameLSTM(self.tok.vocab_size(), args.wemb,
                                                             self.tok.word_to_index['<PAD>'], args.rnn_dim,
                                                             args.dropout).cuda()
            else:
                self.speaker_decoder = model.SpeakerDecoder(self.tok.vocab_size(), args.wemb, self.tok.word_to_index['<PAD>'],
                                                            args.rnn_dim, args.dropout).cuda()
            self.progress_indicator = model.ProgressIndicator().cuda()
            self.matching_network = model.MatchingNetwork().cuda()
            self.feature_predictor = model.FeaturePredictor().cuda()
            self.angle_predictor = model.AnglePredictor().cuda()
            if args.upload:
                speaker_model = get_sync_dir('lyx/snap/speaker/state_dict/best_val_unseen_bleu')
            else:
                speaker_model = os.path.join(args.R2R_Aux_path, 'snap/speaker/state_dict/best_val_unseen_bleu')
            states = torch.load(speaker_model)
            self.speaker_decoder.load_state_dict(states["decoder"]["state_dict"])
            self.aux_models = (self.speaker_decoder, self.progress_indicator, self.matching_network, self.feature_predictor, self.angle_predictor)

            self.aux_optimizer = args.optimizer(
                list(self.speaker_decoder.parameters())
                + list(self.progress_indicator.parameters())
                + list(self.matching_network.parameters())
                + list(self.feature_predictor.parameters())
                + list(self.angle_predictor.parameters())
                , lr=args.lr)
        
            self.all_tuple = [
                ("encoder", self.encoder, self.encoder_optimizer),
                ("decoder", self.decoder, self.decoder_optimizer),
                ("critic", self.critic, self.critic_optimizer),
                ("speaker_decoder", self.speaker_decoder, self.aux_optimizer),
                ("progress_indicator", self.progress_indicator, self.aux_optimizer),
                ("matching_network", self.matching_network, self.aux_optimizer),
                ("feature_predictor", self.feature_predictor, self.aux_optimizer),
                ("angle_predictor", self.angle_predictor, self.aux_optimizer)
            ]
        else:
            self.all_tuple = [
                ("encoder", self.encoder, self.encoder_optimizer),
                ("decoder", self.decoder, self.decoder_optimizer),
                ("critic", self.critic, self.critic_optimizer)
            ]

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        self.bce_loss = nn.BCELoss().cuda()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)


    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.byte().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        if args.sparseObj:
            Obj_leng = [ob['obj_s_feature'].shape[0] for ob in obs]
            if args.catfeat == 'none':
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.instEmb),
                                    dtype=np.float32)
            elif args.denseObj:
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.instEmb),
                                     dtype=np.float32)
            else:
                sparseObj = np.zeros((len(obs), max(Obj_leng), args.instEmb+args.instHE),
                                    dtype=np.float32)
            for i,ob in enumerate(obs):
                sparseObj[i, :Obj_leng[i], :] = ob['obj_s_feature']
        if args.denseObj:
            Obj_leng = [ob['obj_d_feature'].shape[0] for ob in obs]
            if args.catfeat == 'none':
                denseObj = np.zeros((len(obs), max(Obj_leng), args.feature_size),
                                    dtype=np.float32)
            else:
                denseObj = np.zeros((len(obs), max(Obj_leng), args.feature_size+args.angle_feat_size),
                                    dtype=np.float32)
            for i,ob in enumerate(obs):
                denseObj[i, :Obj_leng[i], :] = ob['obj_d_feature']
        features = np.empty((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        if args.sparseObj and not(args.denseObj):
            return Variable(torch.from_numpy(sparseObj), requires_grad=False).cuda(), Obj_leng, Variable(
                torch.from_numpy(features), requires_grad=False).cuda()
        elif args.denseObj and not (args.sparseObj):
            return Variable(torch.from_numpy(denseObj), requires_grad=False).cuda(), Obj_leng, Variable(
                torch.from_numpy(features), requires_grad=False).cuda()
        elif args.denseObj and args.sparseObj:
            return Variable(torch.from_numpy(sparseObj), requires_grad=False).cuda(),Variable(torch.from_numpy(denseObj), requires_grad=False).cuda(),\
                   Obj_leng, Variable(torch.from_numpy(features), requires_grad=False).cuda()
        else:
            return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']                         # Image feat
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()
        candidate_feat, candidate_leng = self._candidate_variable(obs)
        # if args.denseObj and (not args.sparseObj):
        #     denseObj, Obj_leng, features = self._feature_variable(obs)
        #     return input_a_t, denseObj, Obj_leng, features, candidate_feat, candidate_leng
        # else:
        f_t = self._feature_variable(obs)      # Image features from obs
        return input_a_t, f_t, candidate_feat, candidate_leng

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]

        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx_mask = seq_mask

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        v_ctx = [] # ctx before language att
        vl_ctx = [] # ctx after language att
        h1 = h_t
        fea_loss = 0.
        ang_loss = 0.
        for t in range(self.episode_len):
            ObjFeature_mask = None
            sparseObj = None
            denseObj = None
            f_t = None
            Obj_leng = None
            input_a_t, f_t_tuple, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            if args.sparseObj and (not args.denseObj):
                sparseObj, Obj_leng, f_t = f_t_tuple
            elif args.denseObj and (not args.sparseObj):
                denseObj, Obj_leng, f_t = f_t_tuple
            elif args.denseObj and args.sparseObj:
                sparseObj, denseObj, Obj_leng, f_t = f_t_tuple
            else:
                f_t = f_t_tuple

            if Obj_leng is not None:
                ObjFeature_mask = utils.length2mask(Obj_leng)
            if speaker is not None:       # Apply the env drop mask to the feat
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise
                # if args.sparseObj:
                #     if args.catfeat == 'none':
                #         sparseObj *= noise
                #     else:
                #         sparseObj[...,:-args.instHE] *= noise
                # if args.denseObj:
                #     if args.catfeat == 'none':
                #         denseObj *=noise
                #     else:
                #         denseObj[...,:-args.angle_feat_size] *=noise

            h_t, c_t, logit, h1 = self.decoder(input_a_t,candidate_feat,
                                               h1, c_t,
                                               ctx, ctx_mask,feature=f_t,
                                               sparseObj=sparseObj,denseObj=denseObj,
                                               ObjFeature_mask=ObjFeature_mask,
                                               already_dropfeat=(speaker is not None))

            v_ctx.append(h_t)
            vl_ctx.append(h1)
            hidden_states.append(h_t)

            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            candidate_mask = utils.length2mask(candidate_leng)
            if args.submit:     # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = 2.
                        else:                   # Incorrect
                            reward[i] = -2.
                    else:                       # The action is not end
                        reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                        if reward[i] > 0:                           # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        else:
                            raise NameError("The action doesn't change the move")
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            if args.aux_option:
                if args.modfea:
                    _mask = torch.from_numpy(mask).cuda().bool()
                    _mask = ~_mask # 1 invalid, 0 valid
                    target_aux = target.clone()
                    target_aux[_mask] = 0
                    target_aux = target_aux.unsqueeze(1).unsqueeze(2)
                    target_aux = target_aux.expand(-1, 1, candidate_feat.size(2))
                    selected_feat = torch.gather(candidate_feat, 1, target_aux)
                    selected_feat = selected_feat.squeeze() # batch_size, feat_size
                    selected_feat[_mask] = 0
                    feature_label = selected_feat[:, :-args.angle_feat_size]
                    angle_label = selected_feat[:, -4:]
                    feature_pred = self.feature_predictor(h1)
                    angle_pred = self.angle_predictor(h1)
                    if args.mask_fea:
                        fea_loss += torch.mean(F.mse_loss(feature_pred, feature_label, reduce=False) * (~_mask).float().unsqueeze(1))
                        ang_loss += torch.mean(F.mse_loss(angle_pred, angle_label, reduce=False) * (~_mask).float().unsqueeze(1))
                    else:
                        fea_loss += F.mse_loss(feature_pred, feature_label)
                        ang_loss += F.mse_loss(angle_pred, angle_label)
                else:
                    # old style
                    _mask = target == -100
                    target_aux = target.clone()
                    target_aux[_mask] = 0
                    target_aux = target_aux.unsqueeze(1).unsqueeze(2)
                    target_aux = target_aux.expand(-1, 1, candidate_feat.size(2))
                    selected_feat = torch.gather(candidate_feat, 1, target_aux)
                    selected_feat = selected_feat.squeeze() # batch_size, feat_size
                    selected_feat[_mask] = 0
                    feature_label = selected_feat[:, :-args.angle_feat_size]
                    angle_label = selected_feat[:, -4:]
                    feature_pred = self.feature_predictor(h1)
                    angle_pred = self.angle_predictor(h1)
                    fea_loss += F.mse_loss(feature_pred, feature_label)
                    ang_loss += F.mse_loss(angle_pred, angle_label)

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            ObjFeature_mask = None
            sparseObj = None
            denseObj = None
            f_t = None
            Obj_leng = None
            input_a_t, f_t_tuple, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            if args.sparseObj and (not args.denseObj):
                sparseObj, Obj_leng, f_t = f_t_tuple
            elif args.denseObj and (not args.sparseObj):
                denseObj, Obj_leng, f_t = f_t_tuple
            elif args.denseObj and args.sparseObj:
                sparseObj, denseObj, Obj_leng, f_t = f_t_tuple
            else:
                f_t = f_t_tuple
            if Obj_leng is not None:
                ObjFeature_mask = utils.length2mask(Obj_leng)
            if speaker is not None:
                candidate_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise
                # if args.sparseObj:
                #     if args.catfeat == 'none':
                #         sparseObj *= noise
                #     else:
                #         sparseObj[...,-args.instHE] *= noise
                # if args.denseObj:
                #     if args.catfeat == 'none':
                #         denseObj *= noise
                #     else:
                #         denseObj[...,:-args.angle_feat_size] *= noise
            last_h_, _, _, _ = self.decoder(input_a_t,candidate_feat,
                                               h1, c_t,
                                               ctx, ctx_mask,feature=f_t,
                                               sparseObj=sparseObj,denseObj=denseObj,
                                               ObjFeature_mask=ObjFeature_mask,
                                               already_dropfeat=(speaker is not None))
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())
                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.logs['rl_loss'].append(rl_loss.detach())
            self.loss += rl_loss

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.logs['ml_loss'].append(ml_loss.detach())
            self.loss += ml_loss

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.
        
        if args.aux_option:
            # auxiliary tasks
            h_t = h_t.unsqueeze(0)
            c_t = c_t.unsqueeze(0)
            insts = utils.gt_words(perm_obs)

            # if args.modspe:
            #     l = ctx.size(1)
            #     insts = insts[:, :l]
            
            v_ctx = torch.stack(v_ctx, dim=1)
            vl_ctx = torch.stack(vl_ctx, dim=1)
            decode_mask = [torch.tensor(mask) for mask in masks]
            decode_mask = (1 - torch.stack(decode_mask, dim=1)).bool().cuda()  # different definition about mask
            
            # aux #1: speaker recover loss
            eps = 1e-6
            if abs(args.speWeight - 0) > eps:
                if args.modspe:
                    logits = self.speaker_decoder(insts, v_ctx, decode_mask, ctx)#########
                    # logits = logits.permute(0, 2, 1).contiguous()
                    _logits = torch.zeros(logits.size(0), insts.size(1), logits.size(2)).cuda()
                    _logits[:, :logits.size(1), :logits.size(2)] = logits
                    logits = _logits.permute(0, 2, 1).contiguous()
                else:
                    logits, _, _ = self.speaker_decoder(insts, v_ctx, decode_mask, h_t, c_t)
                        # Because the softmax_loss only allow dim-1 to be logit,
                        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
                    logits = logits.permute(0, 2, 1).contiguous()
                spe_loss = self.softmax_loss(
                        input=logits[:, :, :-1],  # -1 for aligning
                        target=insts[:, 1:]  # "1:" to ignore the word <BOS>
                        )
                spe_loss = spe_loss * args.speWeight
                self.loss += spe_loss
                self.logs['spe_loss'].append(spe_loss.detach())
            else:
                self.logs['spe_loss'].append(0)
        
            # aux #2: progress indicator
            if abs(args.proWeight - 0) > eps and not (args.no_train_rl and train_rl):
                if args.modpro:
                    prob = self.progress_indicator(vl_ctx)
                    valid_mask = ~decode_mask
                    progress_label = utils.progress_generator(decode_mask)
                    pro_loss = F.binary_cross_entropy(prob.squeeze(), progress_label, reduce=False)
                    pro_loss = torch.mean(pro_loss * valid_mask.float()) * args.proWeight # mask out
                    self.loss += pro_loss
                    self.logs['pro_loss'].append(pro_loss.detach())
                else:
                    prob = self.progress_indicator(vl_ctx)
                    progress_label = utils.progress_generator(decode_mask)
                    pro_loss = self.bce_loss(prob.squeeze(), progress_label)
                    pro_loss = pro_loss * args.proWeight
                    self.loss += pro_loss
                    self.logs['pro_loss'].append(pro_loss.detach())
            else:
                self.logs['pro_loss'].append(0)
            
            # aux #3: inst matching
            if abs(args.matWeight - 0) > eps and not (args.no_train_rl and train_rl):
                if args.modmat:
                    for i in range(v_ctx.shape[1]):
                        if i == 0:
                            h1 = v_ctx[:, i, :]
                        else:
                            _h1 = v_ctx[:, i, :]
                            valid_mask = ~decode_mask[:, i] # True: move, False: already finished BEFORE THIS ACTION
                            h1 = h1 * (1-valid_mask.float().unsqueeze(1)) + _h1 * valid_mask.float().unsqueeze(1) # update active feature
                    batch_size = h1.shape[0]
                    rand_idx = torch.randperm(batch_size)
                    order_idx = torch.arange(0, batch_size)
                    matching_mask = torch.empty(batch_size).random_(2).bool()
                    same_idx = rand_idx == order_idx
                    label = (matching_mask | same_idx).float().unsqueeze(1).cuda()  # 1 same, 0 different
                    if args.mat_detach:
                        l_ctx = ctx[:,0,:].detach()
                        # l_ctx = torch.cat((ctx[:,0,:], ctx[:,-1,:]), dim=1).detach()
                    else:
                        # l_ctx = torch.cat((ctx[:,0,:], ctx[:,-1,:]), dim=1)
                        l_ctx = ctx[:,0,:]
                    new_h1 = label * h1 + (1 - label) * h1[rand_idx, :]
                    new_l_ctx = label * l_ctx + (1 - label) * l_ctx[rand_idx, :]
                    # vl_pair = torch.cat((new_h1, h1), dim=1)
                    # vl_pair = torch.cat((new_l_ctx, l_ctx), dim=1)
                    prob = self.matching_network(new_h1, l_ctx)
                    prob1 = self.matching_network(new_h1, h1)
                    prob2 = self.matching_network(new_l_ctx, l_ctx)
                    mat_loss = F.binary_cross_entropy(prob, label) * args.matWeight
                    mat_loss1 = F.binary_cross_entropy(prob1, label) * args.matWeight
                    mat_loss2 = F.binary_cross_entropy(prob2, label) * args.matWeight
                    self.loss += mat_loss + mat_loss1 + mat_loss2
                else:
                    h1 = v_ctx[:, -1, :]
                    batch_size = h1.shape[0]
                    rand_idx = torch.randperm(batch_size)
                    order_idx = torch.arange(0, batch_size)
                    matching_mask = torch.empty(batch_size).random_(2).bool()
                    same_idx = rand_idx == order_idx
                    label = (matching_mask | same_idx).float().unsqueeze(1).cuda()  # 1 same, 0 different
                    new_h1 = label * h1 + (1 - label) * h1[rand_idx, :]
                    l_ctx = ctx[:,0,:].detach()
                    vl_pair = torch.cat((new_h1, l_ctx), dim=1)
                    prob = self.matching_network(vl_pair)
                    mat_loss = F.binary_cross_entropy(prob, label) * args.matWeight
                    if args.mat_mask:
                        label_mask = ~decode_mask[:, -1]
                        label_mask = label_mask & label_mask[rand_idx]
                        mat_loss = F.binary_cross_entropy(prob, label, reduce=False) * args.matWeight
                        mat_loss = torch.mean(mat_loss.squeeze() * label_mask.float())
                    self.loss += mat_loss
                self.logs['mat_loss'].append(mat_loss.detach())
            else:
                self.logs['mat_loss'].append(0)
            
            # aux #4: feature prediction
            fea_loss = fea_loss * args.feaWeight
            self.loss += fea_loss
            self.logs['fea_loss'].append(fea_loss.detach())

            # aux #5: angle prediction
            ang_loss = ang_loss * args.angWeight
            self.loss += ang_loss
            self.logs['ang_loss'].append(ang_loss.detach())

        return traj

    def _dijkstra(self):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # Dijk Graph States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)

        # Dijk Algorithm
        for _ in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.env._get_obs()

            # Update the floyd graph
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))

            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            # Run one decoding step
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat,
                                                      h_t, h1, c_t,
                                                      ctx, ctx_mask,
                                                      False)

            # Update the dijk graph's states with the newly visited viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask.bool(), -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item() 
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        """
        self.env.reset()
        results = self._dijkstra()
        """
        return from self._dijkstra()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """

        # Compute the speaker scores:
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.cuda(), can_feats.cuda()
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).cuda()
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
        if args.aux_option:
            for model in self.aux_models:
                model.train()
            self.aux_optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()
        if args.aux_option:
            self.aux_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        for param in self.all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        for param in self.all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1

