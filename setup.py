from setuptools import setup, find_packages

from codecs import open
from os import path
import re


here = path.abspath(path.dirname(__file__))
package_name = 'BDPoisson1D'
version_file = path.join(here, package_name, '_version.py')
with open(version_file, 'rt') as f:
    version_file_line = f.read()
version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(version_re, version_file_line, re.M)
if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in %s.' % (version_file,))

readme_file = path.join(here, 'README.md')
with open(readme_file, encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version=version_string,

    description='BD Finite Difference Poisson equation solver',
    long_description=long_description,

    url='https://github.com/bond-anton/BDPoisson1D',

    author='Anton Bondarenko',
    author_email='bond.anton@gmail.com',

    license='Apache Software License',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='FiniteDifference PDE Poisson',

    packages=find_packages(exclude=['demo', 'tests', 'docs', 'contrib']),
    install_requires=['numpy', 'scipy>=0.17.0', 'matplotlib', 'BDMesh>=0.1.1'],
    dependency_links=['https://github.com/bond-anton/BDMesh/tarball/master#egg=BDMesh-0.1.1'],
    test_suite='nose.collector',
    tests_require=['nose']
)
