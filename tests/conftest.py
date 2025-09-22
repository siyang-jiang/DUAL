"""Test configuration and shared fixtures for DUAL tests."""

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf


@pytest.fixture
def device():
    """Return available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_config():
    """Return dummy configuration for testing."""
    return OmegaConf.create({
        'seed': 42,
        'n_way': 5,
        'n_shot': 1,
        'n_query': 15,
        'backbone': 'resnet12',
        'feature_dim': 640,
        'device': 'cpu',
        'inter_set_alignment': {
            'enabled': True,
            'method': 'feature_alignment',
            'weight': 1.0
        },
        'intra_set_alignment': {
            'enabled': True,
            'method': 'prototype_alignment',
            'weight': 0.5
        }
    })


@pytest.fixture
def dummy_episode_data():
    """Return dummy episode data for testing."""
    n_way, n_shot, n_query = 5, 1, 15
    
    # Support set: n_way * n_shot examples
    support_x = torch.randn(n_way * n_shot, 3, 84, 84)
    support_y = torch.arange(n_way).repeat_interleave(n_shot)
    
    # Query set: n_way * n_query examples  
    query_x = torch.randn(n_way * n_query, 3, 84, 84)
    query_y = torch.arange(n_way).repeat_interleave(n_query)
    
    return {
        'support_x': support_x,
        'support_y': support_y,
        'query_x': query_x,
        'query_y': query_y
    }


@pytest.fixture
def dummy_features():
    """Return dummy feature tensors for testing."""
    return {
        'features': torch.randn(100, 640),
        'labels': torch.randint(0, 5, (100,)),
        'prototypes': torch.randn(5, 640)
    }