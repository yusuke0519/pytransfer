# # -*- coding: utf-8 -*-

from .mnistr import MNISTR, BiasedMNISTR, get_biased_mnistr
from .opportunity import OppG, OppL
from .usc import USC
from .pacs import PACS, BiasedPACS
from .yale_face import YaleFace
from .adult_income import AdultIncome
# from .speech import Speech, BiasedSpeech
from .amazon import Amazon
from .office_c import OfficeC
from .wisdm import WISDM
from .vlcs import VLCS


def get(name):
    return {
        'mnistr': MNISTR,
        'usc': USC,
        'pacs': PACS,
        'oppG': OppG,
        'oppL': OppL,
        'wisdm': WISDM,
        'office_c': OfficeC,
        'amazon': Amazon,
        'vlcs': VLCS,
        'yale': YaleFace,
        'adult': AdultIncome,
    }.get(name)
