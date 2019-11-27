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
import platform

if platform.system() == 'Linux':
    sys.argv.append('--universal')
    sys.argv.append('--plat-name=manylinux1_x86_64')
else:
    sys.argv.append('--universal')
    sys.argv.append('--plat-name=win_amd64')

from setuptools import setup
from setuptools.extension import Extension
kwargs = {'zip_safe': False}
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

data = []
data.append(os.path.join('mxnet', os.path.basename(LIB_PATH[0])))

if platform.system() == 'Linux':
    liblist = [
        '/home/kk/dev/cudnn/lib64/libcudnn.so.7.1.1',
        '/opt/intel/compilers_and_libraries/linux/lib/intel64/libiomp5.so',
        '/opt/intel/mkl/lib/intel64/libmkl_avx2.so',
        '/opt/intel/mkl/lib/intel64/libmkl_core.so',
        '/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so',
        '/opt/intel/mkl/lib/intel64/libmkl_rt.so'
    ]
else:
    liblist = [
        '../../cudnn/bin/cudnn64_7.dll',
        'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler/libiomp5md.dll',
        'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_avx2.dll',
        'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_core.dll',
        'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_intel_thread.dll',
        'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/mkl_rt.dll',
        'E:/proj/dev/mxcproj/deps/opencv4/build/x64/vc15/bin/opencv_ffmpeg410_64.dll',
        'E:/proj/dev/mxcproj/deps/opencv4/build/x64/vc15/bin/opencv_world410.dll',
    ]


for l in liblist:
    shutil.copy(l, os.path.join(CURRENT_DIR, 'mxnet'))
    data.append(os.path.join('mxnet', os.path.basename(l)))

# Try to generate auto-complete code
try:
    from mxnet.base import _generate_op_module_signature
    from mxnet.ndarray.register import _generate_ndarray_function_code
    from mxnet.symbol.register import _generate_symbol_function_code
    _generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
    _generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)
except: # pylint: disable=bare-except
    pass

setup(name='mxnet',
      version=__version__,
      description='MXNet is an ultra-scalable deep learning framework.',
      packages=find_packages(),
      data_files=[('mxnet', data)],
      include_package_data=True,
      url='https://github.com/apache/incubator-mxnet',
      **kwargs)
