# Copyright 2022 The Languini Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import math
import torch
import sys
import sentencepiece as spm
import torch.nn.functional as F
import torch.distributed as dist


internal_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(internal_path)
sys.path.append(internal_path+ '../../../utils')

from tqdm import tqdm
from languini.train_lib import train_utils
from languini.common_lib import debug_utils
from languini.common_lib import common_utils
from languini.common_lib import parallel_utils
from languini.common_lib.debug_utils import check
from languini.train_lib.logger import CustomXAxisScalar
from utils.kernels import centered_kernal_alignment, features_alignment



DEFAULT_CONFIG = {
    "device": None,
    "max_train_steps": None,
    "seq_len": None,
    "max_eval_steps": 50,
    "grad_clip_norm": 0.0,
    "gradient_accumulation_steps": 1,
    "tokens_per_second": 0,

    # logs
    "log_path": None,
    "log_terminal_every": None,     # print avg train loss to terminal 
    "log_metrics_every": None,      # log many train and time metrics      
    "eval_every": 1_000,            # run evaluation and log results
    "log_ckpt_every": 100,          # save model checkpoint
    "log_grads_every": 5_000,       # log gradients, weights, and step sizes to disk
    "log_activations_every": 5_000, # log model activations to disk
}


def evaluate_alignments(config_s, config_t,  student, teacher,  data_source, max_steps, class_frequencies, last_n=-1, print_progress=False, within_batch=False, weigh_by_frequency=False):
    """
    Evaluates the student and teacher cka and fa. 
    
    Args:
        config_s, config_t (Munch): an experiment config.
        student, teacher: PyTorch models.
        data_source: the source for the input and target batches.
        max_step (int): number of batches do process for evaluation.
        last_n (int): evaluate loss on the last_n targets. If last_n is -1 it will evaluate on all targets.
        print_progress (bool): simple terminal log for eval.py to display progress.
    """

    c_s = config_s
    c_t = config_t
    local_bsz = c_s.eval_batch_size // c_s.n_workers

    assert last_n <= c_s.seq_len and last_n != 0, "we cannot eval on the last_n=0 tokens or more tokens than there are in a sequence!"
    assert max_steps == -1 or max_steps > 0, "Maximum number of steps has to be either -1 or a positive value."

    student.eval()
    teacher.eval()
    data_source.reset()

    batch_count = 0
    total = 0
    features_s = []
    features_t = []
    labels = []
    

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=c_s.device.type == "cuda"):
            
            
            # iterate over batches
            if print_progress and max_steps > 0: 
                progress_bar = tqdm(range(max_steps))
            while total <= 5000: 
                if print_progress and max_steps > 0:
                    progress_bar.update(1)
                    progress_bar.refresh()
                elif print_progress and batch_count % 1_000 == 0 and batch_count > 0:
                    parallel_utils.mprint(f"{batch_count:d}")

                try:
                    # take the next training batch
                    batch_x, batch_y, is_padded = next(data_source)
                    batch_x = batch_x[0]
                    batch_y = batch_y[0]
                    bsz, seqlen = batch_x.shape
                except StopIteration:
                    break
                
                # run the forward pass
                if isinstance(student, torch.nn.parallel.DistributedDataParallel):
                    phi_s = student.module.get_features(batch_x)
                    phi_t = teacher.module.get_features(batch_x)

                else:
                    phi_s = student.get_features(batch_x)
                    phi_t = teacher.get_features(batch_x)



                # If last_n is positive we only care about the last_n losses and targets.
                if last_n == -1:
                    last_n = seqlen
                batch_x = batch_x[:, -last_n:]
                check(batch_x, (bsz, last_n))
                phi_s = phi_s[:, -last_n:, :]
                phi_t = phi_t[:, -last_n:, :]
                batch_y = batch_y[:, -last_n:]
                check(batch_y, (bsz, last_n))
                
                # compute loss
                phi_s = phi_s.reshape(bsz * last_n, c_s.h_dim)
                phi_t = phi_t.reshape(bsz * last_n, c_t.h_dim)
                batch_y = batch_y.reshape(bsz * last_n, 1)
                batch_y_one_hot = F.one_hot(batch_y, num_classes=c_s.vocab_size).to(torch.float).reshape(-1, c_s.vocab_size)
                check(batch_y_one_hot, (bsz*last_n, c_s.vocab_size))

                total+=bsz * last_n

                features_s.append(phi_s)
                features_t.append(phi_t)
                labels.append(batch_y_one_hot)
                
                batch_count+=1


            features_s = torch.vstack(features_s).view(-1, c_s.h_dim)
            features_t = torch.vstack(features_t).view(-1, c_t.h_dim)
            labels = torch.vstack(labels).view(-1, c_s.vocab_size)

            class_frequencies1 = class_frequencies + 1
            class_frequencies1_norm = class_frequencies1/class_frequencies1.sum() # frequencies sum to one
            weight = (1/class_frequencies1_norm).view(1, c_s.vocab_size) # 1xC
            weighted_classes = labels * weight # NxC
            samples_weight = weighted_classes.sum(dim=1).view(-1, 1) #N --> we sum over C because there is only one active class per sample
            samples_weight = samples_weight/samples_weight.max() # normalising

            if c_s.h_dim==c_t.h_dim:
                    FA = features_alignment(features_t, features_s)
            else: FA=torch.Tensor([-1]).to(c_s.device)

            if weigh_by_frequency:
                features_s = features_s*samples_weight
                features_t = features_t*samples_weight

            KS = torch.matmul(features_s, features_s.T)
            KT = torch.matmul(features_t, features_t.T)

            KS = KS/torch.max(KS)
            KT = KT/torch.max(KT)

            CKA = centered_kernal_alignment(KS, KT)
            print("CKA", CKA)
        
        parallel_utils.mprint(f'total number of batches processed: {batch_count}')
        dist.all_reduce(CKA, dist.ReduceOp.SUM)
        dist.all_reduce(FA, dist.ReduceOp.SUM)
        total_machines = parallel_utils.WORLD_SIZE # average CKA and FA results



    return CKA.cpu().item()/total_machines, FA.cpu().item()/total_machines

def evaluation(config, model, state, data_source, max_steps, last_n=-1, print_progress=False):
    """
    Evaluates the model on a datasource without gradient updates or extra logs besides loss.
    
    Args:
        config (Munch): an experiment config.
        model: the PyTorch model.
        state: the latest state of the model if it has one (or None to initialise a new one).
        data_source: the source for the input and target batches.
        max_step (int): number of batches do process for evaluation.
        last_n (int): evaluate loss on the last_n targets. If last_n is -1 it will evaluate on all targets.
        print_progress (bool): simple terminal log for eval.py to display progress.
    """
    c = config
    local_bsz = config.eval_batch_size // c.n_workers

    assert last_n <= c.seq_len and last_n != 0, "we cannot eval on the last_n=0 tokens or more tokens than there are in a sequence!"
    assert all(common_utils.flatten(common_utils.traverse(state, func=lambda x: x is None or x.shape[0] == 1))), "all state elements must have batch size 1!"
    assert max_steps == -1 or max_steps > 0, "Maximum number of steps has to be either -1 or a positive value."

    model.eval()
    data_source.reset()

    batch_count = 0
    total_loss = 0
    total_top_k_counts = {}
    total_token_count = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=c.device.type == "cuda"):
            # distribute the given state over the batch-size
            if state is None:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state = model.module.get_init_state(local_bsz, device=c.device)
                else:
                    state = model.get_init_state(local_bsz, device=c.device)
            else:
                # we distribute the same state across all batch elements
                state = common_utils.traverse(state, func=lambda x: torch.concatenate([x] * local_bsz) if x is not None else None)
            
            # iterate over batches
            if print_progress and max_steps > 0: 
                progress_bar = tqdm(range(max_steps))
            while max_steps == -1 or batch_count <= max_steps:
                if print_progress and max_steps > 0:
                    progress_bar.update(1)
                    progress_bar.refresh()
                elif print_progress and batch_count % 1_000 == 0 and batch_count > 0:
                    parallel_utils.mprint(f"{batch_count:d}")

                try:
                    # take the next training batch
                    batch_x, batch_y, is_padded = next(data_source)
                    batch_x = batch_x[0]
                    batch_y = batch_y[0]
                    bsz, seqlen = batch_x.shape
                except StopIteration:
                    break
                
                # run the forward pass
                logits, state = model(batch_x, state, log=None)
                check(logits, (bsz, seqlen, c.vocab_size))

                # If last_n is positive we only care about the last_n losses and targets.
                if last_n == -1:
                    last_n = seqlen
                batch_x = batch_x[:, -last_n:]
                check(batch_x, (bsz, last_n))
                logits = logits[:, -last_n:, :]
                check(logits, (bsz, last_n, c.vocab_size))
                batch_y = batch_y[:, -last_n:]
                check(batch_y, (bsz, last_n))
                
                # compute loss
                logits = logits.reshape(bsz * last_n, c.vocab_size)
                batch_y = batch_y.reshape(bsz * last_n)
                all_losses = F.cross_entropy(input=logits, target=batch_y, reduction='none')
                check(all_losses, (bsz * last_n,))

                # mask losses that are padded (unlike training, evaluation can result in batches with padded batches)
                if is_padded:
                    all_losses = all_losses.masked_fill(batch_x.reshape(-1) == 0, 0.0)
                    token_count = torch.sum(batch_x != 0)
                else:
                    token_count = torch.tensor([batch_x.numel()], device=all_losses.device)

                # compute accuracy for top-1 and top-10
                topk_counts = train_utils.total_correct(logits, batch_y, topk=(1, 10))
                for key in topk_counts.keys():
                    dist.all_reduce(topk_counts[key], dist.ReduceOp.SUM)
                    if key in total_top_k_counts.keys():
                        total_top_k_counts[key] += topk_counts[key].detach().item()
                    else:
                        total_top_k_counts[key] = topk_counts[key].detach().item()

                total_loss += torch.sum(all_losses).reshape((-1,)).detach()
                total_token_count += token_count.reshape((-1,))
                batch_count += 1
        
        parallel_utils.mprint(f'total number of batches processed: {batch_count}')
        dist.all_reduce(total_loss, dist.ReduceOp.SUM) 
        dist.all_reduce(total_token_count, dist.ReduceOp.SUM)

    return total_loss.item(), total_top_k_counts, total_token_count.item(), state


def log_eval_stats(eval_data_source, eval_steps, last_n, sp, logger, device):
    """Counts the number of eval batches and the length in string bytes. Saves these values for later."""
    eval_data_source.reset()
    eval_batches = [(batch_x, batch_y, is_padded) for i, (batch_x, batch_y, is_padded) in enumerate(eval_data_source) if eval_steps == -1 or i < eval_steps]
    batch_count = len(eval_batches)

    micro_batches = eval_data_source.micro_batches
    local_micro_bsz = eval_data_source.bsz // eval_data_source.micro_batches
    seqlen = eval_data_source.seq_len

    # use tensors to count in case it is distributed across accelerators
    token_count = torch.zeros(1, device=device)
    str_length = torch.zeros(1, device=device)
    for (batch_x, batch_y, is_padded) in eval_batches:
        check(batch_x, (micro_batches, local_micro_bsz, seqlen))
        check(batch_y, (micro_batches, local_micro_bsz, seqlen))

        batch_x = batch_x[:, :, -last_n:]
        batch_y = batch_y[:, :, -last_n:]

        # decode targets and measure length
        batch_y = torch.reshape(batch_y, (batch_y.shape[0] * batch_y.shape[1], -1))
        str_lst = sp.decode(batch_y.cpu().tolist())
        for str in str_lst:
            str_length += len(str)

        # count non-padding tokens
        if is_padded:
            token_count += torch.sum(batch_x != 0)
        else:
            token_count += batch_x.numel()
    
    # sum across accelerators
    dist.all_reduce(str_length, dist.ReduceOp.SUM)
    dist.all_reduce(token_count, dist.ReduceOp.SUM)

    # convert to ints
    str_length = int(str_length.cpu().item())
    token_count = int(token_count.cpu().item())
    
    if parallel_utils.is_main_process() and logger:
        logger.log(
            {
                "eval_batches": batch_count,
                "eval_tokens": token_count,
                "eval_bytes": str_length,
            },
            step=None,
        )

        print(f"Eval batches: {batch_count:,}")
        print(f"Eval bytes: {str_length:,}")
    return str_length, batch_count, token_count


class LMTrainer:
    """A language modelling trainer. """
    
    def __init__(self, config, logger, model, opt, train_batches, eval_batches, scheduler=None):
        train_utils.check_config(config, DEFAULT_CONFIG)
        self.c = c = config
        self.logger = logger
        self.model = model.to(config.device)
        self.opt = opt
        self.scheduler = scheduler
        self.train_batches = train_batches
        self.eval_batches = eval_batches
        self.scaler = torch.cuda.amp.GradScaler(enabled=c.device.type == "cuda")

        # log hyperparameters
        train_utils.log_hyperparams(config, self.logger)

        # log total number of weights
        train_utils.print_model_size(self.model, self.c.vocab_size, self.c.h_dim, self.logger)

        # load model weights and state if a checkpoint is provided
        if "checkpoint_path" in self.c.keys() and self.c.checkpoint_path != "":
            self.model, self.curr_state = train_utils.load_checkpoint(model=self.model, path=c.checkpoint_path)
            print(f"Model checkpoint and state loaded from {c.checkpoint_path}")

        # load tokeniser
        self.sp = train_utils.load_tokeniser(config=c)
        assert self.c.vocab_size == self.sp.vocab_size(), f"config vocab size {c.vocab_size} doesn't match tokeniser vocab size {self.sp.vocab_size()}"

        # get number of eval steps and total eval bytes since that will be the same for every evaluation run
        parallel_utils.mprint("Measure evaluation data size ...")
        self.eval_bytes, _, _ = log_eval_stats(eval_data_source=self.eval_batches,
                                               eval_steps=self.c.max_eval_steps,
                                               sp=self.sp,
                                               logger=self.logger,
                                               device=self.c.device,
                                               last_n=self.c.seq_len)

    def train(self):
        c = self.c
        
        # StopWatches to track time spent doing different things
        load_watch = train_utils.StopWatch()        # batch loading
        forward_watch = train_utils.StopWatch()     # forward pass
        backward_watch = train_utils.StopWatch()    # backward pass
        train_watch = train_utils.StopWatch()       # train step
        eval_watch = train_utils.StopWatch()       # evaluation
        total_watch = train_utils.StopWatch()       # total step
        tokens_seen = 0                             # total number of tokens seen
        
        curr_state = None
        total_watch.start()
        for step in range(c.max_train_steps):
            self.model.train()

            # boolean which tracks if during the current step we do some extra logging
            do_grads_log = c.log_grads_every > 0 and step % c.log_grads_every == 0 and step > 0
            do_activations_log = c.log_activations_every > 0 and step % c.log_activations_every == 0 and step > 0

            if not do_grads_log and not do_activations_log:
                # we only track time when we do no extra logging
                train_watch.start()
            
            # load the next training batch
            avg_loss = torch.tensor(0.0, device=c.device)

            load_watch.start()
            total_batch_x, total_batch_y, _ = next(self.train_batches)
            check(total_batch_x, (c.gradient_accumulation_steps, c.train_batch_size // c.gradient_accumulation_steps // c.n_workers, c.seq_len))
            load_watch.pause().count()

            for micro_step in range(c.gradient_accumulation_steps):

                # trick taken from Karpathy's nanoGPT to not sync on every backward call
                self.model.require_backward_grad_sync = (micro_step == c.gradient_accumulation_steps - 1)
                
                # select the current micro batch
                batch_x = total_batch_x[micro_step]
                batch_y = total_batch_y[micro_step]
                check(batch_x, (c.train_batch_size // c.gradient_accumulation_steps // c.n_workers, c.seq_len))
                bsz, seqlen = batch_x.shape
                
                # run forward pass
                with torch.cuda.amp.autocast(enabled=c.device.type == "cuda"):
                    # get initial state
                    if curr_state is None:
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            curr_state = self.model.module.get_init_state(bsz, device=c.device)
                        else:
                            curr_state = self.model.get_init_state(bsz, device=c.device)

                    # track state size
                    state_size = common_utils.get_total_tensor_size(curr_state)
                    
                    # perform the model forward pass with or without activation logs
                    if do_activations_log:
                        logits, curr_state = self.model(batch_x, curr_state, log=(self.logger, step))
                    else:
                        forward_watch.start()
                        logits, curr_state = self.model(batch_x, curr_state, log=None)
                        forward_watch.pause().count()                    
                    check(logits, (bsz, c.seq_len, c.vocab_size))

                    # compute loss
                    logits = logits.reshape(bsz * c.seq_len, c.vocab_size)
                    if do_grads_log:
                        debug_utils.log_stats_and_dist(logits, "Logits", log=(self.logger, step))
                    batch_y = batch_y.reshape(bsz * c.seq_len)
                    micro_avg_loss = F.cross_entropy(input=logits, target=batch_y).reshape((-1,))
                    check(micro_avg_loss, (1,))

                    # keep a sum of the avg_loss of each micro batch
                    avg_loss = avg_loss + micro_avg_loss.detach()
                    
                    # detach state, log, and check state size
                    curr_state = common_utils.traverse(curr_state, func=lambda x: None if x is None else x.detach())
                    if do_grads_log:
                        curr_state_lst = common_utils.flatten(common_utils.traverse(curr_state, func=lambda x: None if x is None else x.cpu()))
                        for state_idx, state in enumerate(curr_state_lst):
                            if not state is None:
                                debug_utils.log_stats_and_dist(state, f"state{state_idx}", log=(self.logger, step))
                    
                    new_state_size = common_utils.get_total_tensor_size(curr_state)
                    if state_size != 0:
                        assert state_size == new_state_size, f"After forward call state size changed from {state_size} to {new_state_size}"

                # scale loss in case of lower precision and perform the backward pass
                backward_watch.start()
                # we need to divide the loss by the number of gradient acc. steps to get the average gradient before we step below
                micro_avg_loss = micro_avg_loss / c.gradient_accumulation_steps
                if self.scaler:
                    self.scaler.scale(micro_avg_loss).backward()
                else:
                    micro_avg_loss.backward()
                backward_watch.pause().count()

            # collect the avg_loss over all micro steps and devices and compute the average
            dist.reduce(avg_loss, op=dist.ReduceOp.SUM, dst=0)
            avg_loss = avg_loss.detach().item() / (parallel_utils.WORLD_SIZE * c.gradient_accumulation_steps)
            
            # unscale gradients before clipping and logging
            if self.scaler:
                self.scaler.unscale_(self.opt)

            # clip gradients if clip is larger than 0.0
            if c.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip_norm)

            # perform an optimiser step and log weights, gradients, and step sizes
            if do_grads_log:
                debug_utils.log_weight_stats(self.model, "Weights", log=(self.logger, step))
                debug_utils.log_gradient_stats(self.model, "Grads", log=(self.logger, step))
                debug_utils.step_and_log_diff(self.scaler, self.opt, self.model, "Step", log=(self.logger, step))
            elif self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            
            # learning rate schedule has to step too
            if self.scheduler:
                self.scheduler.step()

            tokens_seen += c.train_batch_size * c.seq_len
            
            if not do_grads_log and not do_activations_log:
                train_watch.pause().count()

            # log train loss to terminal
            if c.log_terminal_every > 0 and step % c.log_terminal_every == 0 and parallel_utils.is_main_process():
                print(f"step={step:6d}  loss={avg_loss:0.5f}", flush=True)

            # Perfom a validation set evaluation (fixed number of batches)
            if c.eval_every > 0 and step % c.eval_every == 0 and step > 0:
                eval_watch.start()
                self.validation(curr_state=curr_state, step=step)
                eval_watch.pause().count()

            # Write logs to disk
            if parallel_utils.is_main_process() and c.log_metrics_every > 0 and step % c.log_metrics_every == 0 and step > 0:
                
                # log general speed metrics - only do so when we are not doing some extra logging (activations or gradients)
                if not do_grads_log and not do_activations_log:
                    tokens_per_batch = total_batch_x.numel()
                    load_time_per_batch = load_watch.read()
                    forward_time_per_batch = forward_watch.read()
                    backward_time_per_batch = backward_watch.read()
                    step_time_per_batch = train_watch.read()
                    total_time_per_batch = total_watch.read()
                    eval_time = eval_watch.read()
                    tokens_per_second = tokens_per_batch / step_time_per_batch
                    iter_per_second = 1. / step_time_per_batch
                    
                    print(f"tokens per second: {round(tokens_per_second):,}")

                    self.logger.log(
                        {
                            "_time/tokens_per_second": tokens_per_second,
                            "_time/iterations_per_second": iter_per_second,
                            "_time/load_batch": load_time_per_batch,
                            "_time/forward": forward_time_per_batch,
                            "_time/backward": backward_time_per_batch,
                            "_time/train_step": step_time_per_batch,
                            "_time/total_step": total_time_per_batch,
                        },
                        step
                    )

                    if eval_time > 0:
                        self.logger.log({"_time/eval": eval_time}, step)

                self.logger.log(
                    {
                        "_train/loss": avg_loss,
                        "_train/tokens_seen": tokens_seen,
                    },
                    step
                )
                curr_lrs = [pg['lr'] for pg in self.opt.param_groups]
                for idx, lr in enumerate(curr_lrs):  # lr for each param group
                    self.logger.log({f"_train/learning_rate_{idx}": lr}, step)

            # Write the current model weights to disk
            if parallel_utils.is_main_process() and c.log_ckpt_every > 0 and step % c.log_ckpt_every == 0 and step > 0:
                self.save_checkpoint(self.logger, step)

            total_watch.count()
        
        # Final validation run and checkpoint
        self.validation(curr_state=curr_state, step=step)
        if parallel_utils.is_main_process():
            self.save_checkpoint(self.logger, step)

    def validation(self, curr_state, step):
        """Run the model on the test data."""
        c = self.c
        eval_state = common_utils.traverse(curr_state, lambda x: x[:1] if x is not None else None)
        eval_total_loss, eval_total_topk, eval_token_count, _ = evaluation(config=c,
                                                                           model=self.model,
                                                                           state=eval_state,
                                                                           data_source=self.eval_batches,
                                                                           max_steps=c.max_eval_steps)
        # loss and ppl over number of tokens
        eval_avg_loss = eval_total_loss / eval_token_count
        eval_ppl = math.exp(eval_avg_loss)
        # loss and ppl over number of bytes
        eval_norm_loss = eval_total_loss / self.eval_bytes
        eval_norm_ppl = math.exp(eval_norm_loss)
        # accuracy over tokens
        eval_topk_accs = {key: eval_total_topk[key] / eval_token_count for key in eval_total_topk.keys()}

        if parallel_utils.is_main_process():
            number_of_tokens = (step + 1) * c.train_batch_size * c.seq_len # +1 as steps 0-indexed
            theoretical_gpu_seconds = number_of_tokens / c.tokens_per_second if c.tokens_per_second > 0 else 0  
            # Note, you cannot log floating point 'steps' so you cannot compute gpu hours here.
            def log_over_all_axes(name, value):
                """Logs value over steps, tokens, and gpu seconds."""
                metrics = {
                    name: value,
                    f"{name}_over_tokens": CustomXAxisScalar(value, axis_name="n_tokens", axis_val=number_of_tokens),
                    f"{name}_over_gpuseconds": CustomXAxisScalar(value, axis_name="gpu_seconds", axis_val=theoretical_gpu_seconds),
                }
                self.logger.log(metrics, step)

            log_over_all_axes("_eval/normalised_loss", eval_norm_loss)
            self.logger.log(
                {
                    "_eval/loss": eval_avg_loss,
                    "_eval/total_loss": eval_total_loss,
                },
                step
            )

            # skip ppl logging for initial loss which skews the plot unnecessarily
            if eval_ppl < 1_000:
                log_over_all_axes("_eval/ppl", eval_ppl)    
            if eval_norm_ppl < 1_000:
                log_over_all_axes("_eval/normalised_ppl", eval_norm_ppl)
            for key in eval_topk_accs:
                log_over_all_axes(f"_eval/top{key}_acc", eval_topk_accs[key])
            print(f"EVAL step={step:d} loss={eval_avg_loss:0.5f} acc={eval_topk_accs[1]:0.5f}") 

    def save_checkpoint(self, logger, step):
        """Saves a checkpoint of the current model to disk. """

        def _save_checkpoint(path):
            # create folder if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            # remove previous log file there is one        
            file = os.path.join(path, "model.pt")
            if os.path.exists(file):
                os.remove(file)

            # Write checkpoint
            with open(file, 'wb') as f:
                torch.save({
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "opt_state_dict": self.opt.state_dict(),
                    }, f)
            print(f"Checkpoint written at step {step} to:\n{file}")

        if logger.use_tb:
            ckpt_path = os.path.join(logger.log_path, "checkpoints")
            _save_checkpoint(ckpt_path)

        if logger.use_wandb:
            ckpt_path = os.path.join(logger.wandb_run_dir, "checkpoints")
            _save_checkpoint(ckpt_path)


class LMDistilTrainer:
    """A language modelling trainer. """
    
    def __init__(self, config, config_t, logger, model, teacher, opt, train_batches, eval_batches, scheduler=None, class_frequencies=None):
        train_utils.check_config(config, DEFAULT_CONFIG)
        self.c = c = config
        self.c_t =  config_t
        self.logger = logger
        self.model = model.to(config.device)
        self.teacher = teacher.to(config.device)
        self.opt = opt
        self.scheduler = scheduler
        self.train_batches = train_batches
        self.eval_batches = eval_batches
        self.scaler = torch.cuda.amp.GradScaler(enabled=c.device.type == "cuda")
        self.class_frequencies = torch.Tensor(class_frequencies).to(self.c.device)

        # log hyperparameters
        train_utils.log_hyperparams(config, self.logger)

        # log total number of weights
        train_utils.print_model_size(self.model, self.c.vocab_size, self.c.h_dim, self.logger)

        assert "teacher_checkpoint_path" in c.keys() and c.teacher_checkpoint_path != "", "Teacher must be loaded from checkpoint. No checkpoint provided."
        path = os.path.join(c.teacher_checkpoint_path, "checkpoints")
        file = os.path.join(path, "model.pt")
        self.teacher, self.teacher_curr_state = train_utils.load_checkpoint(model=self.teacher, path=file)
        print(f"Teacher checkpoint and state loaded from {c.teacher_checkpoint_path}")


        # load model weights and state if a checkpoint is provided
        if "checkpoint_path" in self.c.keys() and self.c.checkpoint_path != "":
            self.model, self.curr_state = train_utils.load_checkpoint(model=self.model, path=c.checkpoint_path)
            print(f"Model checkpoint and state loaded from {c.checkpoint_path}")

        # load tokeniser #WARN should be same tokeniser as the teacher
        self.sp = train_utils.load_tokeniser(config=c)
        assert self.c.vocab_size == self.sp.vocab_size(), f"config vocab size {c.vocab_size} doesn't match tokeniser vocab size {self.sp.vocab_size()}"


        # get number of eval steps and total eval bytes since that will be the same for every evaluation run
        parallel_utils.mprint("Measure evaluation data size ...")
        self.eval_bytes, _, _ = log_eval_stats(eval_data_source=self.eval_batches,
                                               eval_steps=self.c.max_eval_steps,
                                               sp=self.sp,
                                               logger=self.logger,
                                               device=self.c.device,
                                               last_n=self.c.seq_len)

    def train(self):
        c = self.c
        alpha = c.alpha
        
        # StopWatches to track time spent doing different things
        load_watch = train_utils.StopWatch()        # batch loading
        forward_watch = train_utils.StopWatch()     # forward pass
        backward_watch = train_utils.StopWatch()    # backward pass
        train_watch = train_utils.StopWatch()       # train step
        eval_watch = train_utils.StopWatch()       # evaluation
        total_watch = train_utils.StopWatch()       # total step
        tokens_seen = 0                             # total number of tokens seen

        curr_state = None
        teacher_curr_state = None
        total_watch.start()
        average_logits_magnitude = 0 # running estimate
        total = 0
        T = c.temperature
        print(f"Training with temperature {T}")

        if c.mse: print("Using MSE loss to train.")

        for step in range(c.max_train_steps):
            self.model.train()
            self.teacher.eval()
            step_total = 0

            # boolean which tracks if during the current step we do some extra logging
            do_grads_log = c.log_grads_every > 0 and step % c.log_grads_every == 0 and step > 0
            do_activations_log = c.log_activations_every > 0 and step % c.log_activations_every == 0 and step > 0
            do_alignments_log = c.log_alignments_every > 0 and step % c.log_alignments_every == 0 and step > 0

            if not do_grads_log and not do_activations_log:
                # we only track time when we do no extra logging
                train_watch.start()
            
            # load the next training batch
            avg_loss = torch.tensor(0.0, device=c.device)

            if do_alignments_log: 
                # accumulating features over micro batches
                features_s = []
                features_t = []

            load_watch.start()
            total_batch_x, total_batch_y, _ = next(self.train_batches)
            check(total_batch_x, (c.gradient_accumulation_steps, c.train_batch_size // c.gradient_accumulation_steps // c.n_workers, c.seq_len))
            load_watch.pause().count()

            for micro_step in range(c.gradient_accumulation_steps):

                # trick taken from Karpathy's nanoGPT to not sync on every backward call
                self.model.require_backward_grad_sync = (micro_step == c.gradient_accumulation_steps - 1)
                
                # select the current micro batch
                batch_x = total_batch_x[micro_step]
                batch_y = total_batch_y[micro_step]
                check(batch_x, (c.train_batch_size // c.gradient_accumulation_steps // c.n_workers, c.seq_len))
                bsz, seqlen = batch_x.shape
                
                # run forward pass
                with torch.cuda.amp.autocast(enabled=c.device.type == "cuda"):
                    # get initial state
                    if curr_state is None:
                        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                            curr_state = self.model.module.get_init_state(bsz, device=c.device)
                        else:
                            curr_state = self.model.get_init_state(bsz, device=c.device)

                    #TODO: same state to teacher and student??
                    if teacher_curr_state is None:
                        if isinstance(self.teacher, torch.nn.parallel.DistributedDataParallel):
                            teacher_curr_state = self.teacher.module.get_init_state(bsz, device=c.device)
                        else:
                            teacher_curr_state = self.teacher.get_init_state(bsz, device=c.device)

                    # track state size
                    state_size = common_utils.get_total_tensor_size(curr_state)
                    
                    # perform the model forward pass with or without activation logs
                    if do_activations_log:
                        logits, curr_state = self.model(batch_x, curr_state, log=(self.logger, step))
                    else:
                        forward_watch.start()
                        logits, curr_state = self.model(batch_x, curr_state)
                        forward_watch.pause().count()  
                    
                    t_logits, teacher_curr_state = self.teacher(batch_x, teacher_curr_state)
                    
                
                    check(logits, (bsz, c.seq_len, c.vocab_size))

                    # compute loss
                    logits = logits.reshape(bsz * c.seq_len, c.vocab_size)
                    t_logits = t_logits.reshape(bsz * c.seq_len, c.vocab_size)


                    with torch.no_grad(): 
                        average_non_max = (t_logits.sum(dim=1) - t_logits.max(dim=1)[0])/(c.vocab_size-1) # average over the non-max outputs
                        average_logits_magnitude += (t_logits.max(dim=1)[0] - average_non_max).sum(dim=0) 

                    if do_grads_log:
                        debug_utils.log_stats_and_dist(logits, "Logits", log=(self.logger, step))
                    
                    batch_y = batch_y.reshape(bsz * c.seq_len)
                    batch_y_one_hot = F.one_hot(batch_y, num_classes=c.vocab_size).to(torch.float)
                    check(batch_y_one_hot, (bsz*c.seq_len, c.vocab_size))

                    if c.mse:
                        micro_avg_labels_loss = F.mse_loss(logits, batch_y_one_hot * average_logits_magnitude/total).reshape((-1,)) 
                        micro_avg_distil_loss = F.mse_loss(logits, t_logits).reshape((-1,)) 
                    else:
                        micro_avg_labels_loss = F.cross_entropy(input=logits, target=batch_y).reshape((-1,))#F.mse_loss(logits, F.one_hot(batch_y, num_classes=)).reshape((-1,))#F.cross_entropy(input=logits, target=batch_y).reshape((-1,))
                        micro_avg_distil_loss = F.kl_div(input=F.log_softmax(logits/T, dim=1), target=F.softmax(t_logits/T, dim=1), log_target=False, reduction='none').sum(dim=1).mean().reshape((-1,)) * (T**2) # temperature rescaling (for gradients)

                    check(micro_avg_labels_loss, (1,))
                    check(micro_avg_distil_loss, (1,))
                    micro_avg_loss = alpha*micro_avg_labels_loss+(1-alpha)*micro_avg_distil_loss


                    if do_alignments_log and step_total<5000: 
                        self.model.eval()
                        with torch.no_grad(): 
                            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                                phi_s = self.model.module.get_features(batch_x)   
                                phi_t = self.teacher.module.get_features(batch_x) 
                            else: 
                                phi_s = self.model.get_features(batch_x)
                                phi_t = self.teacher.get_features(batch_x)
                        self.model.train()
                        phi_s = phi_s.reshape(bsz * c.seq_len, c.h_dim)
                        phi_t = phi_t.reshape(bsz * c.seq_len, self.c_t.h_dim)
                        features_s.append(phi_s)
                        features_t.append(phi_t)

                    
                    total+=bsz*c.seq_len
                    step_total+=bsz*c.seq_len

                    # keep a sum of the avg_loss of each micro batch
                    avg_loss = avg_loss + micro_avg_loss.detach()
                    
                    # detach state, log, and check state size
                    curr_state = common_utils.traverse(curr_state, func=lambda x: None if x is None else x.detach())
                    if do_grads_log:
                        curr_state_lst = common_utils.flatten(common_utils.traverse(curr_state, func=lambda x: None if x is None else x.cpu()))
                        for state_idx, state in enumerate(curr_state_lst):
                            if not state is None:
                                debug_utils.log_stats_and_dist(state, f"state{state_idx}", log=(self.logger, step))
                    
                    new_state_size = common_utils.get_total_tensor_size(curr_state)
                    if state_size != 0:
                        assert state_size == new_state_size, f"After forward call state size changed from {state_size} to {new_state_size}"

                # scale loss in case of lower precision and perform the backward pass
                backward_watch.start()
                # we need to divide the loss by the number of gradient acc. steps to get the average gradient before we step below
                micro_avg_loss = micro_avg_loss / c.gradient_accumulation_steps
                if self.scaler:
                    self.scaler.scale(micro_avg_loss).backward()
                else:
                    micro_avg_loss.backward()
                backward_watch.pause().count()


            # collect the avg_loss over all micro steps and devices and compute the average
            dist.reduce(avg_loss, op=dist.ReduceOp.SUM, dst=0)
            avg_loss = avg_loss.detach().item() / (parallel_utils.WORLD_SIZE * c.gradient_accumulation_steps)


            if do_alignments_log:
                features_s = torch.vstack(features_s).view(-1, c.h_dim)
                features_t = torch.vstack(features_t).view(-1, self.c_t.h_dim)
                if c.h_dim==self.c_t.h_dim:
                    FA = features_alignment(features_t, features_s)
                else: FA=torch.Tensor([-1]).to(c.device)
                KS = torch.matmul(features_s, features_s.T)
                KT = torch.matmul(features_t, features_t.T)

                CKA = centered_kernal_alignment(KT,KS)

                dist.reduce(CKA, op=dist.ReduceOp.SUM, dst=0)
                dist.reduce(FA, op=dist.ReduceOp.SUM, dst=0)
                avg_cka = CKA.cpu().item()/parallel_utils.WORLD_SIZE
                avg_fa = FA.cpu().item()/parallel_utils.WORLD_SIZE
            
            # unscale gradients before clipping and logging
            if self.scaler:
                self.scaler.unscale_(self.opt)

            # clip gradients if clip is larger than 0.0
            if c.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip_norm)

            # perform an optimiser step and log weights, gradients, and step sizes
            if do_grads_log:
                debug_utils.log_weight_stats(self.model, "Weights", log=(self.logger, step))
                debug_utils.log_gradient_stats(self.model, "Grads", log=(self.logger, step))
                debug_utils.step_and_log_diff(self.scaler, self.opt, self.model, "Step", log=(self.logger, step))
            elif self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            
            # learning rate schedule has to step too
            if self.scheduler:
                self.scheduler.step()

            tokens_seen += c.train_batch_size * c.seq_len
            
            if not do_grads_log and not do_activations_log:
                train_watch.pause().count()

            # log train loss to terminal
            if c.log_terminal_every > 0 and step % c.log_terminal_every == 0 and parallel_utils.is_main_process():
                print(f"step={step:6d}  loss={avg_loss:0.5f}", flush=True)

            # Perfom a validation set evaluation (fixed number of batches)
            if c.eval_every > 0 and step % c.eval_every == 0 and step > 0:
                eval_watch.start()
                self.validation(curr_state=curr_state, step=step)
                eval_watch.pause().count()

            # Write logs to disk
            if parallel_utils.is_main_process() and c.log_metrics_every > 0 and step % c.log_metrics_every == 0 and step > 0:
                
                # log general speed metrics - only do so when we are not doing some extra logging (activations or gradients)
                if not do_grads_log and not do_activations_log:
                    tokens_per_batch = total_batch_x.numel()
                    load_time_per_batch = load_watch.read()
                    forward_time_per_batch = forward_watch.read()
                    backward_time_per_batch = backward_watch.read()
                    step_time_per_batch = train_watch.read()
                    total_time_per_batch = total_watch.read()
                    eval_time = eval_watch.read()
                    tokens_per_second = tokens_per_batch / step_time_per_batch
                    iter_per_second = 1. / step_time_per_batch
                    
                    print(f"tokens per second: {round(tokens_per_second):,}")

                    self.logger.log(
                        {
                            "_time/tokens_per_second": tokens_per_second,
                            "_time/iterations_per_second": iter_per_second,
                            "_time/load_batch": load_time_per_batch,
                            "_time/forward": forward_time_per_batch,
                            "_time/backward": backward_time_per_batch,
                            "_time/train_step": step_time_per_batch,
                            "_time/total_step": total_time_per_batch,
                        },
                        step
                    )

                    if eval_time > 0:
                        self.logger.log({"_time/eval": eval_time}, step)

                self.logger.log(
                    {
                        "_train/loss": avg_loss,
                        "_train/tokens_seen": tokens_seen
                    },
                    step
                )
                self.logger.log(
                    {
                        "_train/cka":avg_cka,
                        "_train/fa":avg_fa
                    }, 
                    step=step
                )
                curr_lrs = [pg['lr'] for pg in self.opt.param_groups]
                for idx, lr in enumerate(curr_lrs):  # lr for each param group
                    self.logger.log({f"_train/learning_rate_{idx}": lr}, step)

            # Write the current model weights to disk
            if parallel_utils.is_main_process() and c.log_ckpt_every > 0 and step % c.log_ckpt_every == 0 and step > 0:
                self.save_checkpoint(self.logger, step)

            total_watch.count()
        
        # Final validation run and checkpoint
        self.validation(curr_state=curr_state, step=step)
        
        if parallel_utils.is_main_process():
            self.save_checkpoint(self.logger, step)
            #self.log_everything()

    def validation(self, curr_state, step):
        """Run the model on the test data."""
        c = self.c
        eval_state = common_utils.traverse(curr_state, lambda x: x[:1] if x is not None else None)
        eval_total_loss, eval_total_topk, eval_token_count, _ = evaluation(config=c,
                                                                           model=self.model,
                                                                           state=eval_state,
                                                                           data_source=self.eval_batches,
                                                                           max_steps=c.max_eval_steps)
        
        # measure validation cka and fa with teacher
        cka, fa = evaluate_alignments(self.c, self.c_t, 
                                      self.model, self.teacher, 
                                      class_frequencies=self.class_frequencies,
                                      data_source=self.eval_batches,
                                      max_steps=c.max_eval_steps, 
                                      weigh_by_frequency=True)
        
        # loss and ppl over number of tokens
        eval_avg_loss = eval_total_loss / eval_token_count
        eval_ppl = math.exp(eval_avg_loss)
        # loss and ppl over number of bytes
        eval_norm_loss = eval_total_loss / self.eval_bytes
        eval_norm_ppl = math.exp(eval_norm_loss)
        # accuracy over tokens
        eval_topk_accs = {key: eval_total_topk[key] / eval_token_count for key in eval_total_topk.keys()}

        if parallel_utils.is_main_process():
            number_of_tokens = (step + 1) * c.train_batch_size * c.seq_len # +1 as steps 0-indexed
            theoretical_gpu_seconds = number_of_tokens / c.tokens_per_second if c.tokens_per_second > 0 else 0  
            # Note, you cannot log floating point 'steps' so you cannot compute gpu hours here.
            def log_over_all_axes(name, value):
                """Logs value over steps, tokens, and gpu seconds."""
                metrics = {
                    name: value,
                    f"{name}_over_tokens": CustomXAxisScalar(value, axis_name="n_tokens", axis_val=number_of_tokens),
                    f"{name}_over_gpuseconds": CustomXAxisScalar(value, axis_name="gpu_seconds", axis_val=theoretical_gpu_seconds),
                }
                self.logger.log(metrics, step)

            log_over_all_axes("_eval/normalised_loss", eval_norm_loss)
            self.logger.log(
                {
                    "_eval/loss": eval_avg_loss,
                    "_eval/total_loss": eval_total_loss,
                },
                step
            )

            self.logger.log(
                {
                    "_eval/cka":cka,
                    "_eval/fa":fa
                }, 
                step=step
            )

            # skip ppl logging for initial loss which skews the plot unnecessarily
            if eval_ppl < 1_000:
                log_over_all_axes("_eval/ppl", eval_ppl)    
            if eval_norm_ppl < 1_000:
                log_over_all_axes("_eval/normalised_ppl", eval_norm_ppl)
            for key in eval_topk_accs:
                log_over_all_axes(f"_eval/top{key}_acc", eval_topk_accs[key])
            print(f"EVAL step={step:d} loss={eval_avg_loss:0.5f} acc={eval_topk_accs[1]:0.5f}") 

    def save_checkpoint(self, logger, step):
        """Saves a checkpoint of the current model to disk. """

        def _save_checkpoint(path):
            # create folder if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)

            # remove previous log file there is one        
            file = os.path.join(path, "model.pt")
            if os.path.exists(file):
                os.remove(file)

            # Write checkpoint
            with open(file, 'wb') as f:
                torch.save({
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "opt_state_dict": self.opt.state_dict(),
                    }, f)
            print(f"Checkpoint written at step {step} to:\n{file}")

        if logger.use_tb:
            ckpt_path = os.path.join(logger.log_path, "checkpoints")
            _save_checkpoint(ckpt_path)

        if logger.use_wandb:
            ckpt_path = os.path.join(self.c.project_path, self.c.relative_log_path, 'chkpts', self.c.exp_name)
            _save_checkpoint(ckpt_path)

    def log_everything(self, train_time, train_ppl, train_acc, val_ppl, val_acc):
        experiment_log = self.c

        experiment_log['train_time'] = train_time
        experiment_log['final_train_acc_S'] = train_acc
        experiment_log['final_val_acc_S'] = val_acc

        # dumping everything into a log file
        path = self.logger.log_path  
        if not os.path.exists(path): os.makedirs(path)
        with open(path+ "/logs.txt", 'a') as f:
                f.write(json.dumps(experiment_log) + '\n')