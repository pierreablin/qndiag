#! /usr/bin/env python

import os
from setuptools import setup, find_packages

descr = """Joint diagonalization in Python"""

version = None
with open(os.path.join('qndiag', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'qndiag'
DESCRIPTION = descr
MAINTAINER = 'Pierre Ablin'
MAINTAINER_EMAIL = 'pierreablin@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/pierreablin/qndiag.git'
VERSION = version
URL = 'https://github.com/pierreablin/qndiag'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          install_requires=[
              'numpy >=1.14',
          ],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=find_packages(),
          )
