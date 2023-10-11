"""
这个文件用于记录网络训练的各项参数
如 train_epoch，learning_rate，decay_gamma 等
"""
Mode = 'cpu'  # cpu 或 gpu

EEGNet_eeg = {
    'net_args': {
        'channel': 30,
        'time': 400,
        'kernlength': 51,
        'out_dim': 16 * 400 // 100
    },
    'hyper_param': {
        'learning_rate': 0.0004,
        'step_size': 30,
        'gamma': 0.8,
    }
}

EEGNet_nirs = {
    'net_args': {
        'channel': 36,
        'time': 20,
        'kernlength': 5,
        'out_dim': 16 * 20 // 10
    },
    'hyper_param': {
        'learning_rate': 0.0002,
        'step_size': 30,
        'gamma': 0.8,
    }
}

CRNN_eeg = {
    'net_args': {
        'channel': (15, 15),
        'time': 400,
        'kernlength': 51,
        'out_dim': 16
    },
    'hyper_param': {
        'learning_rate': 0.0004,
        'step_size': 30,
        'gamma': 0.5,
    }
}

CRNN_nirs = {
    'net_args': {
        'channel': (15, 15),
        'time': 20,
        'out_dim': 16
    },
    'hyper_param': {
        'learning_rate': 0.0001,
        'step_size': 30,
        'gamma': 0.5,
    }
}

LMF_EEGNet = {
    'eeg_args': EEGNet_eeg['net_args'],
    'nirs_args': EEGNet_nirs['net_args'],
    'lmf_args': {
        'eeg_dim': 16 * 400 // 100,
        'nirs_dim': 16 * 20 // 10,
        'out_dim': 16,
        'rank': 4
    },
    'hyper_param': {
        'learning_rate': 0.0002,
        'step_size': 30,
        'gamma': 0.8,
    }
}

LMF_CRNN = {
    'eeg_args': CRNN_eeg['net_args'],
    'nirs_args': CRNN_nirs['net_args'],
    'lmf_args': {
        'eeg_dim': 16,
        'nirs_dim': 16,
        'out_dim': 16,
        'rank': 4
    },
    'hyper_param': {
        'learning_rate': 0.0002,
        'step_size': 30,
        'gamma': 0.5,
    }
}

MLB_EEGNet = {
    'eeg_args': EEGNet_eeg['net_args'],
    'nirs_args': EEGNet_nirs['net_args'],
    'mlb_args': {
        'eeg_dim': 16 * 400 // 100,
        'nirs_dim': 16 * 20 // 10,
        'emb_dim': 16,
        'out_dim': 16,
    },
    'hyper_param': {
        'learning_rate': 0.0002,
        'step_size': 30,
        'gamma': 0.8,
    }
}

MLB_CRNN = {
    'eeg_args': CRNN_eeg['net_args'],
    'nirs_args': CRNN_nirs['net_args'],
    'mlb_args': {
        'eeg_dim': 16,
        'nirs_dim': 16,
        'emb_dim': 16,
        'out_dim': 16,
    },
    'hyper_param': {
        'learning_rate': 0.0001,
        'step_size': 30,
        'gamma': 0.5,
    }
}

Transformer_eeg = {
    'att_args': {
        'emb_dim': 16,
        'seq_len': 20,
        'size': (20, 16),
        'head': 4,
        'ff_dim': 16,
        'drop': 0.1,
    },
    'eeg_args': {
        'in_channel': 1,
        'channel': 30,
        'time': 400,
        'F1': 8,
        'emb_dim': 16,
        'kern_length': 101,
        'drop': 0.1,
        'data_type': 'eeg'
    },
    'hyper_param': {
        'learning_rate': 0.0004,
        'step_size': 30,
        'gamma': 0.8,
    }
}

Transformer_nirs = {
    'att_args': {
        'emb_dim': 16,
        'seq_len': 20,
        'size': (20, 16),
        'head': 4,
        'ff_dim': 16,
        'drop': 0.1,
    },
    'nirs_args': {
        'in_channel': 2,
        'channel': 36,
        'time': 20,
        'F1': 8,
        'emb_dim': 16,
        'kern_length': 11,
        'drop': 0.1,
        'data_type': 'nirs'
    },
    'hyper_param': {
        'learning_rate': 0.0004,
        'step_size': 30,
        'gamma': 0.8,
    }
}

MulTransformer = {
    'att_args': {
        'emb_dim': 16,
        'seq_len': 20,
        'size': (20, 16),
        'head': 4,
        'ff_dim': 16,
        'drop': 0.25,
    },
    'eeg_args': {
        'in_channel': 1,
        'channel': 30,
        'time': 400,
        'F1': 8,
        'emb_dim': 16,
        'kern_length': 101,
        'drop': 0.25,
        'data_type': 'eeg',
    },
    'nirs_args': {
        'in_channel': 2,
        'channel': 36,
        'time': 20,
        'F1': 8,
        'emb_dim': 16,
        'kern_length': 11,
        'drop': 0.25,
        'data_type': 'nirs'
    },
    'hyper_param': {
        'learning_rate': 0.0004,
        'step_size': 30,
        'gamma': 0.5,
    }
}