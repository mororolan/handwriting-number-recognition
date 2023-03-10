# -*- coding: utf-8 -*-
# @File    : sys_config.py
# @Author  : Robin Lan
# @Time    : 9/3/23 21:26
# @Software: PyCharm
# @Description:

import sys
import keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform


def check_config():
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print(f"SciPy {sp.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")
