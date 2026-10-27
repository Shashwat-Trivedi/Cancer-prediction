"""
Vercel Serverless Function Entry Point
Directs all incoming HTTP requests to the Flask application instance.
"""

import sys
import os

# Add the project root directory to sys.path so modules and pickled assets can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Expose WSGI application callable for Vercel
app = app
