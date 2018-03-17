# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, exec-used
"""Setup mxnet package."""
from __future__ import absolute_import
import os
import sys
import shutil

sys.argv.append('--universal')
sys.argv.append('--plat-name=win_amd64')

from setuptools import setup
from setuptools.extension import Extension
kwargs = {'install_requires': ['numpy<=1.13.3,>=1.8.2', 'requests==2.18.4', 'graphviz==0.8.1'], 'zip_safe': False}
from setuptools import find_packages

# We can not import `mxnet.info.py` in setup.py directly since mxnet/__init__.py
# Will be invoked which introduces dependences
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, '../python/mxnet/libinfo.py')
libinfo = {'__file__': libinfo_py}
exec(compile(open(libinfo_py, "rb").read(), libinfo_py, 'exec'), libinfo, libinfo)

LIB_PATH = libinfo['find_lib_path']()
__version__ = libinfo['__version__']

shutil.rmtree(os.path.join(CURRENT_DIR, 'mxnet'), ignore_errors=True)
shutil.copytree(os.path.join(CURRENT_DIR, '../python/mxnet'),
                os.path.join(CURRENT_DIR, 'mxnet'))

shutil.copy(LIB_PATH[0], os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\cudnn64_7.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\libiomp5md.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\mkl_avx2.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\mkl_core.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\mkl_intel_thread.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\mkl_rt.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\opencv_ffmpeg341_64.dll', os.path.join(CURRENT_DIR, 'mxnet'))
shutil.copy(r'D:\proj\dev\mxaio\bin\opencv_world341.dll', os.path.join(CURRENT_DIR, 'mxnet'))

# Try to generate auto-complete code
try:
    from mxnet.base import _generate_op_module_signature
    from mxnet.ndarray.register import _generate_ndarray_function_code
    from mxnet.symbol.register import _generate_symbol_function_code
    _generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
    _generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)
except: # pylint: disable=bare-except
    print("EXCEPT")
    pass

data = []
data.append(os.path.join('mxnet', os.path.basename(LIB_PATH[0])))
data.append('mxnet/cudnn64_7.dll')
data.append('mxnet/libiomp5md.dll')
data.append('mxnet/mkl_avx2.dll')
data.append('mxnet/mkl_core.dll')
data.append('mxnet/mkl_intel_thread.dll')
data.append('mxnet/mkl_rt.dll')
data.append('mxnet/opencv_ffmpeg341_64.dll')
data.append('mxnet/opencv_world341.dll')

setup(name='mxnet',
      version=__version__,
      description='MXNet is an ultra-scalable deep learning framework.',
      packages=find_packages(),
      data_files=[('mxnet', data)],
      include_package_data=True,
      url='https://github.com/apache/incubator-mxnet',
      **kwargs)
