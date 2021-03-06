import os
import sys
from setuptools import setup

install_requires = [
    "numpy",
    "pandas",
    "loompy",
    "scipy",
    "scanpy",
    "matplotlib",
    "torch"
]


setup(
    name="scClass",
    description="A immune cell classifier tool create by supervised deep learning",
    version='0.9.29',
    url='https://github.com/majaja068/scRNA-CellType-classifier',
    install_requires=install_requires,
    packages = [
        "scClass",
        "scClass.data",
        "scClass.notebook"
    ],
    package_data = {
      "": ["*.md"],
      "scClass":["*","*/*","*/*/*"]
    },
    keywords=[
        "scRNA-seq",
        "cell-type",
        "gene-expression",
        "classifier",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9.6",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ]

)

#### upload command
'''
cd download/scClass
python3 setup.py sdist
tar -ztvf dist/scClass-0.9.x.tar.gz
gdrive upload dist/scClass-0.9.x.tar.gz
'''