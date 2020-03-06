# # -*- coding: utf-8 -*-
from . import mnistr_network
from . import oppG


def get(name):
    return {
        'mnistr': (mnistr_network.Encoder, mnistr_network.Classifier),
        'oppG': (oppG.Encoder, oppG.Classifier),
        'oppL': (oppG.Encoder, oppG.Classifier),
        'usc': (oppG.Encoder, oppG.Classifier),
    }.get(name)
