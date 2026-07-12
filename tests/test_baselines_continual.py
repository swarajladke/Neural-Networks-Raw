"""
Raw AGNIS — tests/test_baselines_continual.py

Unit tests for replay and EWC baseline model wrappers.
"""

import torch
import pytest
from agnis.sequence.sequence_wrapper import (
    RNNReplayBaseline, GRUReplayBaseline, RNNEWCBaseline, GRUEWCBaseline
)


def test_replay_baselines_run():
    vocab_size = 5
    d_hidden = 16
    
    # 1. Initialize
    rnn_replay = RNNReplayBaseline(d_in=vocab_size, d_out=vocab_size, d_hidden=d_hidden, lr=0.01)
    gru_replay = GRUReplayBaseline(d_in=vocab_size, d_out=vocab_size, d_hidden=d_hidden, lr=0.01)
    
    # 2. Train transitions
    x1 = torch.zeros(vocab_size)
    x1[0] = 1.0
    y1 = torch.zeros(vocab_size)
    y1[1] = 1.0
    
    x2 = torch.zeros(vocab_size)
    x2[1] = 1.0
    y2 = torch.zeros(vocab_size)
    y2[2] = 1.0
    
    # Train step
    rnn_replay.reset_sequence_state()
    res1 = rnn_replay.train_transition(x1, y1)
    res2 = rnn_replay.train_transition(x2, y2)
    rnn_replay.reset_sequence_state() # pushes current sequence to replay buffer
    
    gru_replay.reset_sequence_state()
    res3 = gru_replay.train_transition(x1, y1)
    res4 = gru_replay.train_transition(x2, y2)
    gru_replay.reset_sequence_state()
    
    assert "error" in res1
    assert "error" in res3
    assert len(rnn_replay.replay_buffer) == 1
    assert len(gru_replay.replay_buffer) == 1
    assert rnn_replay.replay_buffer[0] == [0, 1]
    
    # 3. Predict transitions
    pred_rnn = rnn_replay.predict_transition(x1)
    pred_gru = gru_replay.predict_transition(x1)
    
    assert pred_rnn.shape == (vocab_size,)
    assert pred_gru.shape == (vocab_size,)
    
    # 4. sleep()
    sleep_rnn = rnn_replay.sleep()
    sleep_gru = gru_replay.sleep()
    
    assert "sleep_mean_loss" in sleep_rnn
    assert "sleep_mean_loss" in sleep_gru


def test_ewc_baselines_run():
    vocab_size = 5
    d_hidden = 16
    
    # 1. Initialize
    rnn_ewc = RNNEWCBaseline(d_in=vocab_size, d_out=vocab_size, d_hidden=d_hidden, lr=0.01, ewc_lambda=50.0)
    gru_ewc = GRUEWCBaseline(d_in=vocab_size, d_out=vocab_size, d_hidden=d_hidden, lr=0.01, ewc_lambda=50.0)
    
    # 2. Train transitions
    x1 = torch.zeros(vocab_size)
    x1[0] = 1.0
    y1 = torch.zeros(vocab_size)
    y1[1] = 1.0
    
    x2 = torch.zeros(vocab_size)
    x2[1] = 1.0
    y2 = torch.zeros(vocab_size)
    y2[2] = 1.0
    
    # First task (task 0)
    rnn_ewc.start_task(0)
    rnn_ewc.reset_sequence_state()
    rnn_ewc.train_transition(x1, y1)
    rnn_ewc.train_transition(x2, y2)
    rnn_ewc.reset_sequence_state()
    
    # Transition to task 1: should compute EWC Fisher data for task 0
    rnn_ewc.start_task(1)
    
    assert len(rnn_ewc.task_optimal_params) == 1
    assert len(rnn_ewc.fishers) == 1
    assert "rnn.weight_ih" in rnn_ewc.task_optimal_params[0]
    assert rnn_ewc.fishers[0]["rnn.weight_ih"].shape == rnn_ewc.rnn.weight_ih.shape
    
    # Train on task 1 with EWC penalty active
    res_ewc = rnn_ewc.train_transition(x1, y1)
    assert "ewc_penalty" in res_ewc
    
    # GRU EWC check
    gru_ewc.start_task(0)
    gru_ewc.reset_sequence_state()
    gru_ewc.train_transition(x1, y1)
    gru_ewc.train_transition(x2, y2)
    gru_ewc.reset_sequence_state()
    gru_ewc.start_task(1)
    
    assert len(gru_ewc.task_optimal_params) == 1
    assert len(gru_ewc.fishers) == 1
