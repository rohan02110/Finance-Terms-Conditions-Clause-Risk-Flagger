"""
Smoke tests for Phase 0 setup.
"""
import pytest
from hello_world import hello_world


def test_hello_world():
    result = hello_world()
    assert "Environment Ready" in result


def test_imports():
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import streamlit
    import pdfplumber

    assert np is not None
    assert pd is not None
    assert sklearn is not None
    assert torch is not None
    assert streamlit is not None
    assert pdfplumber is not None


def test_pytorch_tensor_operation():
    import torch
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = x + y
    assert z.tolist() == [5.0, 7.0, 9.0]
