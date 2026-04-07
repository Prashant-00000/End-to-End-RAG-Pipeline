"""RAG application package."""

# Suppress transformers library warnings about missing optional dependencies
import warnings
import os
import sys

# Filter transformers and torch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='.*transformers.*')
warnings.filterwarnings('ignore', message='.*No module named.*torchvision.*')
warnings.filterwarnings('ignore', message='.*No module named.*timm.*')
warnings.filterwarnings('ignore', message='.*Accessing `__path__`.*')
warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Also suppress at environment level by setting PYTHONWARNINGS
if 'PYTHONWARNINGS' not in os.environ:
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning,ignore::UserWarning'
