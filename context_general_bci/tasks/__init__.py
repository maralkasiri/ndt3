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
class ExperimentalTask(OrderedEnum):
    nlb_maze = "nlb_maze"
    nlb_rtt = "nlb_rtt"
    churchland_maze = "churchland_maze"
    churchland_misc = "churchland_misc"
    odoherty_rtt = "odoherty_rtt"
    dyer_co = "dyer_co"
    gallego_co = "gallego_co"
    pitt_co = "pitt_co"
    observation = "observation"
    ortho = "ortho"
    fbc = "fbc"
    unstructured = "unstructured" # Pitt free play
    delay_reach = "delay_reach"

    marino_batista_mp_bci = "marino_batista_mp_bci"
    marino_batista_mp_reaching = "marino_batista_mp_reaching"
    marino_batista_mp_iso_force = "marino_batista_mp_iso_force"
    cst = 'cst'
    miller = "miller"
    rouse = "rouse"
    chase = "chase"
    mayo = "mayo"
    flint = "flint"
    schwartz = "schwartz"
    perich = "perich"
    hatsopoulos = "hatsopoulos"
    limblab = "limblab"
    mender_fingerctx = "mender_fingerctx"
    deo = "deo"

    falcon_h1 = "falcon_h1"
    falcon_h2 = "falcon_h2"
    falcon_m1 = "falcon_m1"
    falcon_m2 = "falcon_m2"

    mock_half_falcon_m1 = "mock_half_falcon_m1"

    generalized_click = "generalized_click"

from .task_registry import ExperimentalTaskRegistry, ExperimentalTaskLoader
# Exports
from .nlb import MazeLoader, RTTLoader
from .rtt import ODohertyRTTLoader
from .maze import ChurchlandMazeLoader
from .myow_co import DyerCOLoader
from .gallego_co import GallegoCOLoader
from .churchland_misc import ChurchlandMiscLoader
from .pitt_co import PittCOLoader
from .delay_reach import DelayReachLoader
from .marino_batista import MarinoBatistaLoader
from .miller import MillerLoader
from .rouse import RouseLoader
from .chase import ChaseLoader
from .mayo import MayoLoader
from .flint import FlintLoader
from .schwartz_ez import SchwartzLoader
from .cst import CSTLoader
from .hatsopoulos import HatsopoulosLoader
from .mender import MenderLoader
from .limblab import LimbLabLoader
from .deo import DeoLoader
from .falcon import FalconLoader
from .nwb_base import NWBLoader
from .gc_nwb_loader import GCNWBLoader