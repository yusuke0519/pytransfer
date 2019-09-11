# TL_utils
the utilities for the transfer learning

## Directories
- datasets
- learner
- models (currently not supported)

## Versions
- Python 2.7.13 (anaconda2-4.4.0)
- pytorch 0.3.* (You have to manually install pytorch when CUDA version problems occur)

## Datasets
- mnistr.py: rotated MNIST
  - https://openreview.net/forum?id=r1Dx7fbCW
  - the link for the gz data: https://github.com/ghif/mtae/blob/master/mtae.py
- pacs.py: image dataset covering photo, sketch, cartoon and painting domains
  - the link for the paper info and dataset: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
- speech.py: Spoken word recognition across users
  - the link for tensorflow tutorials: https://www.tensorflow.org/tutorials/audio_recognition

## Instructions
- prepare anaconda2-4.4.0
- ```pip install -e .```
- ```apt-get install unrar``` (for Office+Caltech dataset)
