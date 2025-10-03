# from enum import Enum
from ordered_enum import OrderedEnum

r"""
    Dependency notes:
    - We define the enum so there's typing available outside of the tasks module
    - The individual loaders must depend on registry so `register` works
    - The loader registry must depend on the enum so it can be queried
    - To avoid cyclical dependency we must make enum declared before individual loaders
        -  i.e. loader names must be defined in enum rather than enum pulling from loader
"""

class SubjectName(OrderedEnum):
    # We refer to names instead of classes to make converting to singleton pattern easier
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    PTest = "PTest"
    BMI01 = "BMI01"
    BCI02 = "BCI02"
    BCI03 = "BCI03"
    t5 = "T5"
    jenkins = "Jenkins"
    indy = "Indy"
    loco = "Loco"
    nitschke = "Nitschke"
    mihi = "Mihi"
    greyson = "Greyson"
    spike = "Spike"
    chewie = "Chewie"
    han = "Han"
    lando = "Lando"
    reggie = "Reggie"
    earl = "Earl"
    nigel = "Nigel"
    rocky = "Rocky"
    jango = "Jango"
    rouse_p = "Rouse_P"
    rouse_q = "Rouse_Q"
    rouse_a = "Rouse_A"
    rouse_b = "Rouse_B"
    maestro = "Maestro"
    nigel_schwartz = "Nigel_Schwartz"
    rocky_schwartz = "Rocky_Schwartz"

    falcon_h1 = "FALCONH1"
    falcon_m1 = "FALCONM1"
    falcon_h2 = "FALCONH2"
    falcon_m2 = "FALCONM2"

    perich_c = "Perich_C"
    perich_j = "Perich_J"
    perich_m = "Perich_M"
    perich_t = "Perich_T"
    batista_f = "Batista_F"
    batista_e = "Batista_E"

    hat_generic = 'Hat_Generic'
    chestek_generic = 'Chestek_Generic'
    limblab_generic = 'LimbLab_Generic'

    brnbciP2 = 'brnbciP2'
    brnbciP3 = 'brnbciP3'

from .array_info import SubjectInfo, ArrayInfo, ArrayID, GeometricArrayInfo, AliasArrayInfo, SortedArrayInfo
from .array_registry import SubjectArrayRegistry, create_spike_payload
# ? Should we be referencing this instance or the class in calls? IDK
subject_array_registry = SubjectArrayRegistry()

# These import lines ensure registration
from . import pitt_chicago
from . import nlb_monkeys

