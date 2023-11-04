from setuptools import setup, find_packages
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name='DeepMIL',
    version='0.0.1',
    author='Hassan Bassiouny, Ricardo Fleck, Augustin Krause',
    author_email='',
    description="Implementation of the Machine Learning Model described in the paper 'Attention-based Deep Multiple Instance Learning'",
    url='https://github.com/augustinkrause/Attention-based-Deep-Multiple-Instance-Learning',
    packages=find_packages(exclude=['tests']),
    keywords='',
    install_requires=[
        'torch',
        'numpy'
    ],
    python_requires='>=3.7'
)
