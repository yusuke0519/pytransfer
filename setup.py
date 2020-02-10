from setuptools import setup, find_packages

# currently torch==0.3 is supported
requires = ['numpy',
            'tensorflow==1.15.2',
            'memoize==1.0.0',
            'luigi==2.6.2',
            'wget==3.2',
            'torch==0.3.*',
            'torchvision==0.2.1',
            'tqdm==4.23.2',
            'sacred==0.7.4',
            'gitpython==2.1.10',
            'tensorboardX==1.2',
            'rarfile',
            'pandas',
            'matplotlib',
            'seaborn',
            'scikit-learn']

setup(
    name='pytransfer',
    version='0.0.1',
    description='Dataset and Learner Package for Transfer Learning with PyTorch',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=requires
)
