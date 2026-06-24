# -*- coding: utf-8 -*-
"""Pytest config — make the project root importable."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
