"""RAG application package."""

# Configure logging BEFORE importing transformers
import logging
import os

# Suppress transformers library logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.models').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# Set environment variables to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
