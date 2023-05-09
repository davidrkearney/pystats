from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys


setup(
    name="pystats",
    version="0.1.0",
    author="David Raymond",
    author_email="david.raymond.kearney@gmail.com",
    description="A short description of your package",
    packages=["pystats"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "mypy",
        "tqdm",
        "statsmodels",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)


