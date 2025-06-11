# Information about the real world.
# Includes experimental notes, in lieu of readme
# Ideally, this class can be used outside of this specific codebase.

import os
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import functools
import logging

logger = logging.getLogger(__name__)

from .context_info import (
    ContextInfo,
    ReachingContextInfo,
    RTTContextInfo,
    DyerCOContextInfo,
    GallegoCOContextInfo,
    GDrivePathContextInfo,
    BCIContextInfo,
    BatistaContextInfo,
    MillerContextInfo,
    RouseContextInfo,
    FlintContextInfo,
    ChaseContextInfo,
    MayoContextInfo,
    SchwartzContextInfo,
    FalconContextInfo,
    DANDIContextInfo,
    NHContextInfo,
    MenderContextInfo,
    LimbLabContextInfo,
    DeoContextInfo,
    MooreContextInfo,
)

from context_general_bci.tasks import ExperimentalTask

CLOSED_LOOP_DIR = 'closed_loop_tests'
r"""
    ContextInfo class is an interface for storing meta-information needed by several consumers, mainly the model, that may not be logged in data from various sources.
    ContextRegistry allows consumers to query for this information from various identifying sources.
    Note - external registry calls should use the instance, not the class.
    This appears to be necessary for typing to work more reliably (unclear).

    Binds subject, task, and other metadata like datapath together.
    JY's current view of  ideal dependency tree is
    Consumer (e.g. model training loop) -- depends -- > ContextRegistry --> Task -> Subject Registry
    - (But currently consumer will dive into some task-specific details, loader could be refactored..)

    Querying examples:
    # context = context_registry.query(alias='odoherty_rtt-Loco')[0] (see formatting of alias in `context_info` defns)
    # datapath = './data/odoherty_rtt/indy_20160407_02.mat'
    # context = context_registry.query_by_datapath(datapath)
"""

r"""
    To support a new task
    - Add a new enum value to ExperimentalTask
    - Add experimental config to DatasetConfig
    - Implement a loader (see examples in tasks/)
    - Subclass ContextInfo and implement the abstract methods (register the datapaths)
"""

class ContextRegistry:
    instance = None
    _registry: Dict[str, ContextInfo] = {}
    search_index = None  # allow multikey querying

    def __new__(cls, init_items: List[ContextInfo]=[]):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.search_index = pd.DataFrame()
            cls.instance.register(init_items)
        return cls.instance

    def build_search_index(self, items: List[ContextInfo]):
        index = [{
            'id': item.id,
            'task': item.task,
            'datapath': item.datapath.resolve(),
            **item.get_search_index()
        } for item in items]
        return pd.DataFrame(index)

    def clear(self):
        self._registry = {}
        self.search_index = pd.DataFrame()

    # ! Note, current pattern is to put all experiments in a big list below; not use this register handle.
    def register(self, context_info: List[ContextInfo | None]):
        filter_info = [item for item in context_info if item is not None]
        # Check collision / re-registry
        novel_info = []
        for item in filter_info:
            if self.query_by_id(item.id):
                logger.warning(f"Re-registering {item.id}")
            else:
                novel_info.append(item)
        self.search_index = pd.concat([self.search_index, self.build_search_index(novel_info)])
        for item in filter_info:
            self._registry[item.id] = item

    def query(self, **search) -> Union[ContextInfo, List[ContextInfo], None]:
        def search_query(df):
            if 'alias' not in df:
                print("no alias, nothing registered?")
                return None # test time
            non_str_search = [k for k in search if k != 'alias']
            if non_str_search:
                result = functools.reduce(lambda a, b: a & b, [df[k] == search[k] for k in non_str_search])
            else:
                result = pd.Series(True, index=df.index)
            if 'alias' not in search:
                return result
            return result & df['alias'].str.contains(search['alias'])
        queried = self.search_index.loc[search_query]
        if len(queried) == 0:
            return None
        elif len(queried) > 1:
            out = [self._registry[id] for id in queried['id']]
            return sorted(out)
        return self._registry[queried['id'].values[0]]

    def query_by_datapath(self, datapath: Union[Path, str]) -> ContextInfo:
        if not isinstance(datapath, Path):
            datapath = Path(datapath)
        found = self.search_index[self.search_index.datapath == datapath.resolve()]
        assert len(found) == 1, f"Found {len(found)} matches for {datapath}"
        return self._registry[found.iloc[0]['id']]

    def query_by_id(self, id: str) -> ContextInfo | None:
        if id not in self._registry:
            return None
        return self._registry[id]

# singleton
context_registry = ContextRegistry()

if not os.getenv('NDT_SUPPRESS_DEFAULT_REGISTRY', False):
    # Note we register closed loop dir here as well, maybe don't want that
    context_registry.register([
        *RTTContextInfo.build_several('./data/odoherty_rtt/', alias_prefix='odoherty_rtt'),
        *RTTContextInfo.build_several('./data/heldout_odoherty_rtt/', alias_prefix='heldout_odoherty_rtt'), # Try not to have this anywhere but RNEL-n0. It's a dup of calib/eval split, but not preproc-ed for condition analysis

        # DyerCOContextInfo.build(('mihi', 1), ExperimentalTask.dyer_co, alias='dyer_co_mihi_1'),
        # DyerCOContextInfo.build(('mihi', 2), ExperimentalTask.dyer_co, alias='dyer_co_mihi_2'),
        DyerCOContextInfo.build(('chewie', 1), ExperimentalTask.dyer_co, alias='dyer_co_chewie_1'),
        DyerCOContextInfo.build(('chewie', 2), ExperimentalTask.dyer_co, alias='dyer_co_chewie_2'),

        *GallegoCOContextInfo.build_from_dir('./data/gallego_co', task=ExperimentalTask.gallego_co),
        # *GallegoCOContextInfo.build_from_dir('./data/gallego_co/dedup', task=ExperimentalTask.gallego_co),

        ReachingContextInfo.build('./data/nlb/000128/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze'),
        ReachingContextInfo.build('./data/nlb/000138/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_large'),
        ReachingContextInfo.build('./data/nlb/000139/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_medium'),
        ReachingContextInfo.build('./data/nlb/000140/sub-Jenkins', ExperimentalTask.nlb_maze, alias='mc_maze_small'),
        ReachingContextInfo.build('./data/nlb/000129/sub-Indy', ExperimentalTask.nlb_rtt, alias='mc_rtt'),

        *ReachingContextInfo.build_several('./data/churchland_reaching/000070/sub-Jenkins', ExperimentalTask.churchland_maze, alias_prefix='churchland_maze_jenkins'),
        *ReachingContextInfo.build_several('./data/churchland_reaching/000070/sub-Nitschke', ExperimentalTask.churchland_maze, alias_prefix='churchland_maze_nitschke'),

        # *ReachingContextInfo.build_several('./data/even_chen_delay/000121/sub-JenkinsC', ExperimentalTask.delay_reach, alias_prefix='even_chen_delay_jenkins'),
        *ReachingContextInfo.build_several('./data/delay_reach/000121/sub-Reggie', ExperimentalTask.delay_reach, alias_prefix='even_chen_delay_reggie'),

        # *GDrivePathContextInfo.build_from_dir('./data/churchland_misc', blacklist=[]),
        *GDrivePathContextInfo.build_from_dir('./data/churchland_misc', blacklist=['reggie']),

        # *BCIContextInfo.build_from_nested_dir(f'./data/{CLOSED_LOOP_DIR}', task_map={}, alias_prefix='closed_loop_'), # each dataset deposits into its own session folder
        # *BCIContextInfo.build_from_nested_dir(f'./data/{CLOSED_LOOP_DIR}_outpost', task_map={}, alias_prefix='closed_loop_outpost_'), # each dataset deposits into its own session folder

        # *BCIContextInfo.build_from_dir(f'./data/pitt_broad', task_map={}, alias_prefix='pitt_broad_'),
        # *BCIContextInfo.build_from_dir(f'./data/pitt_bmi01_raw', task_map={}, alias_prefix='pitt_broad_'), # spike times
        # *BCIContextInfo.build_from_dir(f'./data/pitt_bmi01', task_map={}, alias_prefix='pitt_broad_'), # 30ms
        # *BCIContextInfo.build_from_dir(f'./data/chicago_human', task_map={}, alias_prefix='chicago_human_'),
        # *BCIContextInfo.build_from_dir(f'./data/pitt_parity', task_map={}, alias_prefix='parity_'), # For offline analysis, special preprocessing


        # *MillerContextInfo.build_from_dir('./data/miller/adversarial', task=ExperimentalTask.miller),

        *MillerContextInfo.build_from_dir('./data/miller/Jango_ISO_2015', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Spike_ISO_2012', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Chewie_CO_2016', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Mihili_CO_2014', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Mihili_RT_2013_2014', task=ExperimentalTask.miller),
        *MillerContextInfo.build_from_dir('./data/miller/Greyson_Key_2019', task=ExperimentalTask.miller),

        # *MillerContextInfo.build_from_dir('./data/miller/Pop_PG_2021', task=ExperimentalTask.miller), # TODO fixup - it's at 50ms, maybe not worth in near future.

        *RouseContextInfo.build_from_dir('./data/rouse_precision/monk_p/COT_SpikesCombined', task=ExperimentalTask.rouse),
        *RouseContextInfo.build_from_dir('./data/rouse_precision/monk_q/COT_SpikesCombined', task=ExperimentalTask.rouse),
        *RouseContextInfo.build_from_dir('./data/rouse/', task=ExperimentalTask.rouse, is_ksu=True),

        *FlintContextInfo.build_from_dir('./data/flint/', task=ExperimentalTask.flint),
        *ChaseContextInfo.build_from_dir('./data/chase/', task=ExperimentalTask.chase),
        *MayoContextInfo.build_from_dir('./data/mayo/', task=ExperimentalTask.mayo),
        *SchwartzContextInfo.build_from_dir('./data/schwartz/MonkeyN/', task=ExperimentalTask.schwartz),
        *SchwartzContextInfo.build_from_dir('./data/schwartz/MonkeyR/', task=ExperimentalTask.schwartz),

        *DANDIContextInfo.build_from_dir('./data/perich/000688/sub-C', task=ExperimentalTask.perich),
        *DANDIContextInfo.build_from_dir('./data/perich/000688/sub-M', task=ExperimentalTask.perich),
        *DANDIContextInfo.build_from_dir('./data/perich/000688/sub-J', task=ExperimentalTask.perich),
        *DANDIContextInfo.build_from_dir('./data/perich/000688/sub-T', task=ExperimentalTask.perich),

        #  === Eval Sets ===

        *FalconContextInfo.build_from_dir('./data/falcon/000954/sub-HumanPitt-held-in-calib', task=ExperimentalTask.falcon_h1), # , suffix='calib', is_dandi=False),
        *FalconContextInfo.build_from_dir('./data/falcon/000954/sub-HumanPitt-held-out-calib', task=ExperimentalTask.falcon_h1), # , suffix='calib', is_dandi=False),

        *FalconContextInfo.build_from_dir('./data/falcon/000941/sub-MonkeyL-held-in-calib', task=ExperimentalTask.falcon_m1),
        *FalconContextInfo.build_from_dir('./data/falcon/000941/sub-MonkeyL-held-out-calib', task=ExperimentalTask.falcon_m1),

        *FalconContextInfo.build_from_dir('./data/falcon/000950/sub-T5-held-in-calib', task=ExperimentalTask.falcon_h2),
        *FalconContextInfo.build_from_dir('./data/falcon/000950/sub-T5-held-out-calib', task=ExperimentalTask.falcon_h2),

        *FalconContextInfo.build_from_dir('./data/falcon/000953/sub-MonkeyN-held-in-calib', task=ExperimentalTask.falcon_m2),
        *FalconContextInfo.build_from_dir('./data/falcon/000953/sub-MonkeyN-held-out-calib', task=ExperimentalTask.falcon_m2),

        # Just be a little careful about aliases here
        # *RTTContextInfo.build_several('./data/archive/odoherty_rtt', alias_prefix='ARCHIVE_rtt_ARCHIVE'),
        # *BCIContextInfo.build_from_dir('./data/archive/pitt_co', task_map={}, alias_prefix='ARCHIVE_pitt_ARCHIVE_'),
        # *BatistaContextInfo.build_from_dir('./data/archive/cst', alias_prefix='ARCHIVE_cst_ARCHIVE', task=ExperimentalTask.cst),
        # *RTTContextInfo.build_several('./data/archive/odoherty_s1rtt', alias_prefix='ARCHIVE_s1rtt_ARCHIVE'),

        # Primary eval block
        *RTTContextInfo.build_preproc('./data/calib/odoherty_rtt/', alias_prefix='calib_odoherty_calib_rtt'),
        *RTTContextInfo.build_preproc('./data/calib/s1rtt/', alias_prefix='calib_s1rtt_calib_rtt', arrays=['S1']),
        # *BCIContextInfo.build_preproc('./data/calib/pitt_co', alias_prefix='calib_pitt_calib_broad_'),
        # *BCIContextInfo.build_preproc('./data/calib/pitt_co_trialized', alias_prefix='calib_pitt_trialized_broad_'),  # Dup for analysis
        # *BCIContextInfo.build_preproc('./data/calib/pitt_grasp', alias_prefix='calib_pitt_grasp_'), # GB's data
        *BatistaContextInfo.build_from_dir('./data/calib/cst', task=ExperimentalTask.cst, alias_prefix='calib_cst_calib', preproc=True),

        # # Be careful - these shouldn't be trained on
        *RTTContextInfo.build_preproc('./data/eval/odoherty_rtt/', alias_prefix='eval_odoherty_eval_rtt'),
        *RTTContextInfo.build_preproc('./data/eval/s1rtt/', alias_prefix='eval_s1rtt', arrays=['S1']),
        # *BCIContextInfo.build_preproc('./data/eval/pitt_co/', alias_prefix='eval_pitt_eval_broad_'),
        # *BCIContextInfo.build_preproc('./data/eval/pitt_co_trialized/', alias_prefix='eval_pitt_trialized_broad_'),
        # *BCIContextInfo.build_preproc('./data/eval/pitt_grasp', alias_prefix='eval_pitt_grasp_'),
        *BCIContextInfo.build_preproc('./data/eval/chicago_grasp', alias_prefix='eval_chicago_grasp_'),
        *BatistaContextInfo.build_from_dir('./data/eval/cst', task=ExperimentalTask.cst, alias_prefix='eval_cst_eval', preproc=True),

        *BatistaContextInfo.build_from_dir('./data/cst', task=ExperimentalTask.cst),
        *BatistaContextInfo.build_from_dir('./data/marino_batista/earl_multi_posture_isometric_force', task=ExperimentalTask.marino_batista_mp_iso_force),
        *BatistaContextInfo.build_from_dir('./data/marino_batista/earl_multi_posture_dco_reaching', task=ExperimentalTask.marino_batista_mp_reaching),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/earl_multi_posture_bci', task=ExperimentalTask.marino_batista_mp_bci),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/nigel_multi_posture_bci', task=ExperimentalTask.marino_batista_mp_bci),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/nigel_multi_posture_dco_reaching', task=ExperimentalTask.marino_batista_mp_reaching),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/rocky_multi_posture_bci', task=ExperimentalTask.marino_batista_mp_bci),
        # *BatistaContextInfo.build_from_dir('./data/marino_batista/rocky_multi_posture_dco_reaching', task=ExperimentalTask.marino_batista_mp_reaching),

        *NHContextInfo.build_from_nested_dir('./data/hatlab/CO/Velma', task=ExperimentalTask.hatsopoulos),

        *NHContextInfo.build_from_nested_dir('./data/hatlab/Lester/2015', task=ExperimentalTask.hatsopoulos),
        *NHContextInfo.build_from_nested_dir('./data/hatlab/Lester/2016', task=ExperimentalTask.hatsopoulos),
        *NHContextInfo.build_from_nested_dir('./data/hatlab/Hermes/', task=ExperimentalTask.hatsopoulos),
        *NHContextInfo.build_from_nested_dir('./data/hatlab/Jim/', task=ExperimentalTask.hatsopoulos),
        *NHContextInfo.build_from_nested_dir('./data/hatlab/Theseus/2021', task=ExperimentalTask.hatsopoulos),
        *NHContextInfo.build_from_nested_dir('./data/hatlab/Theseus/2022', task=ExperimentalTask.hatsopoulos),

        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Fish', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Greyson', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Jaco', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Jango', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Keedoo', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Kevin', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Pedro', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Spike', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Thelonius', task=ExperimentalTask.limblab),
        *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Thor', task=ExperimentalTask.limblab),

        # Otherwise archive
        # Breaux is not useful atm - no neural data
        # *NHContextInfo.build_from_dir('./data/hatlab/Breaux/2017', task=ExperimentalTask.hatsopoulos),
        # *NHContextInfo.build_from_dir('./data/hatlab/Breaux/2018', task=ExperimentalTask.hatsopoulos),
        # *NHContextInfo.build_from_dir('./data/hatlab/Breaux/2019', task=ExperimentalTask.hatsopoulos),
        # *NHContextInfo.build_from_dir('./data/hatlab/Breaux/2020', task=ExperimentalTask.hatsopoulos),
        # *NHContextInfo.build_from_dir('./data/hatlab/Breaux/2021', task=ExperimentalTask.hatsopoulos),
        # *BCIContextInfo.build_preproc('./data/calib/chicago_grasp', alias_prefix='calib_chicago_grasp_'), # EO's data

        # *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Chewie', task=ExperimentalTask.limblab),
        # *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Mihili', task=ExperimentalTask.limblab), # No relevant data left
        # *LimbLabContextInfo.build_from_nested_dir('./data/limblab/MrT', task=ExperimentalTask.limblab), # Nothing survived preproc
        # *LimbLabContextInfo.build_from_nested_dir('./data/limblab/Pop', task=ExperimentalTask.limblab), # Wireless, sensitive data

        # Analysis
        *BCIContextInfo.build_from_dir(f'./data/pitt_heli_block', task_map={}, alias_prefix='pitt_intra_session_'),
        *MenderContextInfo.build_from_file(f'./data/mender_fingerctx/monkeyN_1D_0.mat', task=ExperimentalTask.mender_fingerctx),
        *MenderContextInfo.build_from_file(f'./data/mender_fingerctx/monkeyN_1D_1.mat', task=ExperimentalTask.mender_fingerctx),
        *MenderContextInfo.build_from_file(f'./data/mender_fingerctx/monkeyN_1D_2.mat', task=ExperimentalTask.mender_fingerctx),
        *DeoContextInfo.build_from_dir(f'./data/deo', task=ExperimentalTask.deo),
        # *MooreContextInfo.build_from_dir(f'./data/moore', task=ExperimentalTask.deo),
    ])
else:
    context_registry.register([
        *BCIContextInfo.build_from_nested_dir(f'./data/{CLOSED_LOOP_DIR}', task_map={}, alias_prefix='closed_loop_'), # each dataset deposits into its own session folder
    ])