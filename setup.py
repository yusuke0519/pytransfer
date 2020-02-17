from setuptools import setup, find_packages

# currently torch==0.3 is supported
requires = []

setup(
    name='pytransfer',
    version='0.0.1',
    description='Dataset and Learner Package for Transfer Learning with PyTorch',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requires
)