import abc
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
import logging

from context_general_bci.config import DatasetConfig
from context_general_bci.subjects import SubjectArrayRegistry, SubjectInfo, SubjectName
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import crop_subject_handles

# FYI: Inherited dataclasses don't call parent's __init__ by default. This is a known issue/feature:
# https://bugs.python.org/issue43835
logger = logging.getLogger(__name__)

# Onnx requires 3.9, kw_only was added in 3.10. We patch with this suggestion https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses/53085935#53085935
@dataclass
class _ContextInfoBase:
    subject: SubjectInfo # note this is an object/value
    task: ExperimentalTask # while this is an enum/key, currently

@dataclass
class _ContextInfoDefaultsBase:
    _arrays: List[str] = field(default_factory=lambda: []) # arrays (without subject handles) that were active in this context. Defaults to all known arrays for subject
    datapath: Path = Path("fake_path") # path to raws - to be provided by subclass (not defaulting to None for typing)
    alias: str = ""


# Regress for py 3.9 compat
# @dataclass(kw_only=True)
@dataclass
class ContextInfo(_ContextInfoDefaultsBase, _ContextInfoBase):
    r"""
        Base (abstract) class for static info for a given dataset.
        Subclasses should specify identity and datapath
    """
    # Context items - this info should be provided in all datasets.

    # These should be provided as denoted in SubjectArrayRegistry WITHOUT subject specific handles.
    # Dropping subject handles is intended to be a convenience since these contexts are essentially metadata management. TODO add some safety in case we decide to start adding handles explicitly as well...

    def __init__(self,
        subject: SubjectInfo,
        task: str,
        _arrays: List[str] = [],
        alias: str = "",
        **kwargs
    ):
        self.subject = subject
        self.task = task
        self.alias = alias
        # This is more or less an abstract method; not ever intended to be run directly.

        # self.build_task(**kwargs) # This call is meaningless since base class __init__ isn't called
        # Task-specific info are responsible for assigning self.datapath

    def __post_init__(self):
        if not self._arrays: # Default to all arrays
            self._arrays = self.subject.arrays.keys()
        else:
            assert all([self.subject.has_array(a) for a in self._arrays]), \
                f"An array in {self._arrays} not found in SubjectArrayRegistry"
        assert self.datapath is not Path("fake_path"), "ContextInfo didn't initialize with datapath"
        if not self.datapath.exists():
            logging.warning(f"ContextInfo datapath not found ({self.datapath})")

    @property
    def array(self) -> List[str]:
        r"""
            We wrap the regular array ID with the subject so we don't confuse arrays across subjects.
            These IDs will be used to query for array geometry later. `array_registry` should symmetrically register these IDs.
        """
        return [self.subject.wrap_array(a) for a in self._arrays]

    @property
    def id(self):
        return f"{self.task}-{self.subject.name.value}-{self._id()}"

    @abc.abstractmethod
    def _id(self):
        raise NotImplementedError

    @property
    def session_embed_id(self):
        return self.id

    @classmethod
    @abc.abstractmethod
    def build_task(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def build_task(cls, **kwargs):
        raise NotImplementedError

    def get_search_index(self):
        # Optional method for allowing searching the registry with these keys
        return {
            'alias': self.alias,
            'subject': self.subject
        }

    def load(self, cfg: DatasetConfig, cache_root: Path):
        loader = ExperimentalTaskRegistry.get_loader(self.task)
        logger.info(f"Preprocessing {self.task}: {self.datapath}...")
        return loader.load(
            self.datapath,
            cfg=cfg,
            cache_root=cache_root,
            subject=self.subject,
            context_arrays=self.array,
            dataset_alias=self.alias,
            task=self.task
        )

    # For sorting
    def __eq__(self, other):
        if not isinstance(other, ContextInfo):
            return NotImplemented
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, ContextInfo):
            return NotImplemented
        return self.id < other.id

    def __gt__(self, other):
        if not isinstance(other, ContextInfo):
            return NotImplemented
        return self.id > other.id

@dataclass
class _ReachingContextInfoBase:
    session: int

@dataclass
class ReachingContextInfo(ContextInfo, _ReachingContextInfoBase):

    def _id(self):
        return f"{self.session}-{self.alias}" # All reaching data get alias

    @classmethod
    def build(cls, datapath_str: str, task: ExperimentalTask, alias: str="", arrays=["main"]):
        datapath = Path(datapath_str)
        if not datapath.exists():
            logger.warning(f"Datapath not found, skipping ({datapath})")
            return None
        subject = SubjectArrayRegistry.query_by_subject(
            datapath.name.split('-')[-1].lower()
        )
        session = int(datapath.parent.name)
        return ReachingContextInfo(
            subject=subject,
            task=task,
            _arrays=arrays,
            alias=alias,
            session=session,
            datapath=datapath,
        )

    @classmethod
    def build_several(cls, datapath_folder_str: str, task: ExperimentalTask, alias_prefix: str = "", arrays=["PMd", "M1"]):
        # designed around churchland reaching data
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder not found, skipping ({datapath_folder})")
            return []
        subject = SubjectArrayRegistry.query_by_subject(
            datapath_folder.name.split('-')[-1].lower()
        )
        session = int(datapath_folder.parent.name)
        all_info = []
        for i, path in enumerate(datapath_folder.glob("*.nwb")):
            alias = f"{alias_prefix}-{path.stem}" if alias_prefix else f"reaching-{subject.name}-{path.stem}"
            all_info.append(ReachingContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=alias,
                session=session,
                datapath=path,
            ))
        return all_info

    def get_search_index(self):
        return {
            **super().get_search_index(),
            'session': self.session,
        }

@dataclass
class GDrivePathContextInfo(ContextInfo):
    # for churchland_misc
    def _id(self):
        return f"{self.datapath}"

    @classmethod
    def build_from_dir(cls, datapath_folder_str: str, blacklist: List[str]=[]):
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder not found, skipping ({datapath_folder})")
            return []
        all_info = []
        for path in datapath_folder.glob("*.mat"):
            subject = path.stem.split('-')[0]
            if subject in blacklist:
                continue
            if subject == 'nitschke':
                arrays = ['PMd', 'M1']
            elif subject == 'jenkins':
                arrays = ['PMd', 'M1']
            elif subject == 'reggie':
                arrays = ['PMd', 'M1']
            # find pre-registered path
            all_info.append(GDrivePathContextInfo(
                subject=SubjectArrayRegistry.query_by_subject(subject),
                task=ExperimentalTask.churchland_misc,
                _arrays=arrays,
                datapath=path,
                alias=f'churchland_misc_{path.stem}',
            ))
        return all_info


DYER_CO_FILENAMES = {
    ('mihi', 1): 'full-mihi-03032014',
    ('mihi', 2): 'full-mihi-03062014',
    ('chewie', 1): 'full-chewie-10032013',
    ('chewie', 2): 'full-chewie-12192013',
}
@dataclass
class DyerCOContextInfo(ReachingContextInfo):
    @classmethod
    def build(cls, handle, task: ExperimentalTask, alias: str="", arrays=["main"], root='./data/dyer_co/'):
        datapath = Path(root) / f'{DYER_CO_FILENAMES[handle]}.mat'
        if not datapath.exists():
            logger.warning(f"Datapath not found, skipping ({datapath})")
            return None
        subject = SubjectArrayRegistry.query_by_subject(
            datapath.name.split('-')[-2].lower()
        )
        session = int(datapath.stem.split('-')[-1])
        return DyerCOContextInfo(
            subject=subject,
            task=task,
            _arrays=arrays,
            alias=alias,
            session=session,
            datapath=datapath,
        )

# Data has been replaced with M1 only data
# GALLEGO_ARRAY_MAP = {
#     'Lando': ['LeftS1Area2'],
#     'Hans': ['LeftS1Area2'],
#     'Chewie': ['M1', 'PMd'], # left hemisphere M1
#     'Mihi': ['M1', 'PMd'],
# }

# CHEWIE_ONLY_M1 = [ # right hemisphere M1. We don't make a separate distinction
#     'Chewie_CO_20150313',
#     'Chewie_CO_20150630',
#     'Chewie_CO_20150319',
#     'Chewie_CO_20150629',
# ]

@dataclass
class GallegoCOContextInfo(ReachingContextInfo):
    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        def make_info(datapath: Path):
            alias = f'{task.value}_{datapath.stem}'
            if alias.endswith('_M1'):
                alias = alias[:-3]
            subject, _, date, *rest = datapath.stem.split('_') # task is CO always
            subject = subject.lower()
            if subject == "mihili":
                subject = "mihi" # alias
            subject = SubjectArrayRegistry.query_by_subject(subject)
            session = int(date)
            if subject.name == SubjectName.mihi and session in [20140303, 20140306]: # in Dyer release
                return None
            arrays = ['M1']
            # arrays = GALLEGO_ARRAY_MAP.get(subject.name.value)
            # if alias in CHEWIE_ONLY_M1:
                # arrays = ['M1']
            return GallegoCOContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=alias,
                session=int(date),
                datapath=datapath,
            )
        infos = map(make_info, Path(root).glob("*.mat"))
        return filter(lambda x: x is not None, infos)


@dataclass
class BCIContextInfo(ReachingContextInfo):
    session_set: int = 0

    # def session_embed_id(self):
    #     return f"{self.session}" # Many overlapping sessions from the same day, preserve ID.

    @classmethod
    def build_from_dir_varied(cls, root: str, task_map: Dict[str, ExperimentalTask], arrays=["main"]):
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        def make_info(datapath: Path):
            if datapath.is_dir():
                alias = datapath.name
                subject, _, session = alias.split('.')
                session_set = 0
                session_type = pitt_metadata.get(alias, 'default')
            else: # matlab file
                alias = datapath.stem
                subject, _, session, _, session_set, _, *session_type = alias.split('_')
                session_type = '_'.join(session_type)
                blacklist_check_key = f'{subject}_session_{session}_set_{session_set}'
                if blacklist_check_key in pitt_metadata:
                    session_type = pitt_metadata[blacklist_check_key]
            subject = crop_subject_handles(subject)
            alias = f'{task_map.get(session_type, ExperimentalTask.unstructured).value}_{alias}'
            # print(f"registering {alias} with type {session_type}, {task_map.get(session_type)}")
            return BCIContextInfo(
                subject=SubjectArrayRegistry.query_by_subject(subject),
                task=task_map.get(session_type, ExperimentalTask.unstructured),
                _arrays=[
                    'lateral_s1', 'medial_s1',
                    'lateral_m1', 'medial_m1',
                ],
                alias=alias,
                session=int(session),
                datapath=datapath,
                session_set=session_set
            )
        infos = map(make_info, Path(root).glob("*"))
        return filter(lambda x: x is not None, infos)

    @classmethod
    def make_info(cls, datapath: Path, task_map: Dict = {}, alias_prefix="", simple=False):
        if datapath.is_dir():
            alias = datapath.name
            subject, _, session = alias.split('.')
            session_set = 0
            session_type = pitt_metadata.get(alias, 'default')
        elif datapath.suffix == '.mat': # matlab file
            alias = datapath.stem
            pieces = alias.split('_')
            pieces = list(filter(lambda x: x != '', pieces))
            if len(pieces) == 5:
                # broad pull
                subject, _, session, _, session_set = pieces
                alias = f'{alias_prefix}{ExperimentalTask.pitt_co.value}_{subject}_{session}_{session_set}'
                task = ExperimentalTask.pitt_co
                # Note we now include location in alias
            else:
                subject, _, session, _, session_set, _, *session_type, control = pieces
                session_type = '_'.join(session_type)
                task = None
                blacklist_check_key = f'{subject}_session_{session}_set_{session_set}'
                if blacklist_check_key in pitt_metadata:
                    session_type = pitt_metadata[blacklist_check_key]
                    control = 'default'
                subject = subject[:3].upper() + subject[3:]
                if simple:
                    alias = f'{alias_prefix}{task_map.get(control, ExperimentalTask.pitt_co).value}_{subject}_{session}_{session_set}'
                    task = task_map.get(control, task_map.get('default', ExperimentalTask.unstructured))
                else:
                    alias = f'{alias_prefix}{task_map.get(control, ExperimentalTask.pitt_co).value}_{subject}_{session}_{session_set}_{session_type}'
                    if any(i in session_type for i in ['2d_cursor_center', '2d_cursor_pursuit', '2d+click_cursor_pursuit']) or alias_prefix == 'pitt_misc_':
                        task = task_map.get(control, task_map.get('default', ExperimentalTask.unstructured))
                    else:
                        task = task_map.get('default', ExperimentalTask.unstructured)
        else:
            return None # Ignore
        arrs = [
            'lateral_s1', 'medial_s1',
            'lateral_m1', 'medial_m1',
        ] if 'BMI01' not in subject else ['lateral_m1', 'medial_m1']
        return BCIContextInfo(
            subject=SubjectArrayRegistry.query_by_subject(subject),
            task=task,
            _arrays=arrs,
            alias=alias,
            session=int(session),
            session_set=int(session_set),
            datapath=datapath,
        )

    @classmethod
    def canonical_path_to_alias(cls, canonical_filename: str):
        # in: <Subject>_session_<Session>_set_<Set>
        # out <Subject>_<Session>_<Set>
        subject, _, session, _, session_set = canonical_filename.split('_')
        return f'{subject}_{session}_{session_set}'

    @classmethod
    def build_from_dir(cls, root: str, task_map: Dict[str, ExperimentalTask], alias_prefix='', simple=False) -> List['BCIContextInfo']:
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        infos = map(lambda x: cls.make_info(x, task_map=task_map, alias_prefix=alias_prefix, simple=simple), Path(root).glob("*"))
        return filter(lambda x: x is not None, infos)

    @classmethod
    def build_from_nested_dir(cls, root: str, task_map: Dict[str, ExperimentalTask], alias_prefix='', simple=False):
        if not Path(root).exists():
            logger.warning(f"Datapath not found, skipping ({root})")
            return []
        for path in Path(root).glob("*"):
            if path.is_dir() and 'CRS' in path.name:
                infos = map(lambda x: cls.make_info(x, task_map=task_map, alias_prefix=alias_prefix, simple=simple), path.glob("*"))
                yield from filter(lambda x: x is not None, infos)

    @classmethod
    def build_preproc(cls, datapath_folder_str: str, alias_prefix: str = ""):
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder {datapath_folder} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            alias = path.stem
            pieces = alias.split('_')
            subject, _, session, _, session_set, *_ = pieces
            task = ExperimentalTask.pitt_co
            alias = f'{alias_prefix}{task.value}_{subject}_{session}_{session_set}'
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return BCIContextInfo(
                subject=subject,
                task=task,
                _arrays=[
                    'lateral_s1', 'medial_s1',
                    'lateral_m1', 'medial_m1',
                ],
                alias=alias,
                session=int(session),
                session_set=int(session_set),
                datapath=path,
            )
        return map(make_info, datapath_folder.glob("*.pth"))

# Not all have S1 - JY would prefer registry to always be right rather than detecting this post-hoc during loading
# So we do a pre-sweep and log down which sessions have which arrays here
RTT_SESSION_ARRAYS = {
    'indy_20160624_03': ['M1', 'M1_all'],
    'indy_20161007_02': ['M1', 'M1_all'],
    'indy_20160921_01': ['M1', 'M1_all'],
    'indy_20170123_02': ['M1', 'M1_all'],
    'indy_20160627_01': ['M1', 'M1_all'],
    'indy_20160927_06': ['M1', 'M1_all'],
    'indy_20161212_02': ['M1', 'M1_all'],
    'indy_20161011_03': ['M1', 'M1_all'],
    'indy_20161026_03': ['M1', 'M1_all'],
    'indy_20161206_02': ['M1', 'M1_all'],
    'indy_20161013_03': ['M1', 'M1_all'],
    'indy_20170131_02': ['M1', 'M1_all'],
    'indy_20160930_02': ['M1', 'M1_all'],
    'indy_20160930_05': ['M1', 'M1_all'],
    'indy_20161024_03': ['M1', 'M1_all'],
    'indy_20170124_01': ['M1', 'M1_all'],
    'indy_20161017_02': ['M1', 'M1_all'],
    'indy_20161027_03': ['M1', 'M1_all'],
    'indy_20160630_01': ['M1', 'M1_all'],
    'indy_20161025_04': ['M1', 'M1_all'],
    'indy_20161207_02': ['M1', 'M1_all'],
    'indy_20161220_02': ['M1', 'M1_all'],
    'indy_20161006_02': ['M1', 'M1_all'],
    'indy_20160915_01': ['M1', 'M1_all'],
    'indy_20160622_01': ['M1', 'M1_all'],
    'indy_20161005_06': ['M1', 'M1_all'],
    'indy_20161014_04': ['M1', 'M1_all'],
    'indy_20160927_04': ['M1', 'M1_all'],
    'indy_20160916_01': ['M1', 'M1_all'],
    'indy_20170127_03': ['M1', 'M1_all'],
}


@dataclass
class _RTTContextInfoBase:
    date_hash: str

@dataclass
class RTTContextInfo(ContextInfo, _RTTContextInfoBase):
    r"""
        We make this separate from regular ReachingContextInfo as subject hash isn't unique enough.
    """

    def _id(self):
        return f"{self.date_hash}-{self._arrays}"

    @classmethod
    def build_several(cls, datapath_folder_str: str, arrays=["M1", "M1_all", "S1"], alias_prefix="rtt"):
        r"""
            TODO: not obvious how we can detect whether datapath has S1 or not
        """
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder {datapath_folder} does not exist. Skipping.")
            return []

        def make_info(path: Path):
            subject, date, _set = path.stem.split("_")
            subject = SubjectArrayRegistry.query_by_subject(subject)
            date_hash = f"{date}_{_set}"
            _arrays = RTT_SESSION_ARRAYS.get(path.stem, arrays)
            return RTTContextInfo(
                subject=subject,
                task=ExperimentalTask.odoherty_rtt,
                _arrays=_arrays,
                alias=f"{alias_prefix}-{subject.name.value}-{date_hash}",
                date_hash=date_hash,
                datapath=path,
            )
        return map(make_info, datapath_folder.glob("*.mat"))


    @classmethod
    def build_preproc(cls, datapath_folder_str: str, alias_prefix: str = "", arrays=["M1"]):
        r"""
            For preprocessed splits produced by dataloaders and `split_eval`. Pytorch already.
        """
        datapath_folder = Path(datapath_folder_str)
        if not datapath_folder.exists():
            logger.warning(f"Datapath folder {datapath_folder} does not exist. Skipping.")
            return []

        def make_info(path: Path):
            subject, date, *tail = path.stem.split("_")
            tail = '_'.join(tail)
            subject = SubjectArrayRegistry.query_by_subject(subject)
            date_hash = f"{date}_{tail}"
            return RTTContextInfo(
                subject=subject,
                task=ExperimentalTask.odoherty_rtt,
                _arrays=arrays,
                alias=f"{alias_prefix}-{subject.name.value}-{date_hash}",
                date_hash=date_hash,
                datapath=path,
            )
        return map(make_info, datapath_folder.glob("*.pth"))


@dataclass
class BatistaContextInfo(ContextInfo):

    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"], alias_prefix="batista", preproc=False):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            if task == ExperimentalTask.cst:
                # Ford_20180627_COCST_TD.mat
                # subject, *_ = root.stem.split("_") # old marino datasets apparently used this?
                subject, *_ = path.stem.split("_")
                if subject == 'Ford':
                    subject = 'batista_f'
                elif subject == 'Earl':
                    subject = 'batista_e'
                arrays = ['main']
            else:
                # data/marino_batista/earl_multi_posture_dco_reaching/DelayedCenterOut_E20210710.mat
                subject = path.parent.name.split('_')[0] # e.g. earl
                arrays = ['M1']
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return BatistaContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{alias_prefix}-{subject.name.value}-{path.stem}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.pth" if preproc else "*.mat"))
        return filter(lambda x: x is not None, infos)

@dataclass
class MillerContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["M1"]):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            subject, *_ = path.stem.split("_")
            subject = subject.lower()
            if subject == "mihili":
                subject = "mihi" # alias
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return MillerContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"miller_{subject.name.value}-{path.stem}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.mat"))
        return filter(lambda x: x is not None, infos)

@dataclass
class RouseContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["M1"], is_ksu=False):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # Format: Q_Spikes_20180418-data.mat
            subject, _, timestamp = path.stem.split("_")
            timestamp = timestamp.split("-")[0]
            subject = f'rouse_{subject.lower()}'
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return RouseContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{'rouse_ksu' if is_ksu else 'rouse'}_{subject.name.value}-{timestamp}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.mat"))
        return filter(lambda x: x is not None, infos)

@dataclass
class ChaseContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["M1"]):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # Format: Rocky_20220216_processed.mat
            subject, timestamp, _ = path.stem.split("_")
            timestamp = timestamp.split("-")[0]
            subject = SubjectArrayRegistry.query_by_subject(subject.lower())
            return ChaseContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{'chase'}_{subject.name.value}-{timestamp}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.mat"))
        return filter(lambda x: x is not None, infos)


@dataclass
class MayoContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["FEF"]):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # Format: combinedMaestroPlxFEF(pa16dir4A).mat
            subject = "Maestro"
            session = path.stem.split("pa")[1].split("dir")[0]
            subject = SubjectArrayRegistry.query_by_subject(subject.lower())
            return MayoContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{'mayo'}_{subject.name.value}-{session}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.mat"))
        return filter(lambda x: x is not None, infos)

@dataclass
class MenderContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_file(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        # Shockingly puts multiple sessions in one run. wcyd.
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            subject = SubjectArrayRegistry.query_by_subject('chestek_generic')
            return MenderContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{'mender'}_{subject.name.value}_{path.stem}",
                datapath=path,
            )
        return [make_info(root)]

@dataclass
class SchwartzContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, _root: str, task: ExperimentalTask, arrays=["M1"]):
        root = Path(_root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # Format: .../MonkeyN/MONKEY_NAME.EZ.SESSION:05d
            if not path.is_dir():
                return None
            subject, _, session = path.name.split('.')
            session = int(session)
            subject = SubjectArrayRegistry.query_by_subject(subject.lower())
            return SchwartzContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{'schwartz'}_{subject.name.value}-{session}",
                datapath=path,
            )
        infos = map(make_info, sorted(list(root.glob("*"))))
        return filter(lambda x: x is not None, infos)


@dataclass
class FlintContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # Format: Flint_2012_e1.mat
            _, year, id_no = path.stem.split("_")
            subject = f'chewie' # inferred monkey c from miller lab in 2010s as chewie
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return FlintContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"flint_{subject.name.value}-{id_no}",
                datapath=path,
            )
        infos = map(make_info, root.glob("*.mat"))
        return filter(lambda x: x is not None, infos)


# For FALCON alias to session key, for FALCON/when it's essential to get session ID, not alias ID.
def explicit_session_reduction(alias: str) -> str:
    if alias.endswith('_calib'):
        alias = alias[:-6]
    if alias.endswith('_eval'):
        alias = alias[:-5]
    if alias.endswith('_minival'):
        alias = alias[:-8]
    if '_set_' in alias: # H1
        alias = alias.split('_set_')[0]
    if '_run_' in alias: # M2
        alias = alias.split('_run_')[0]
    if '-Run' in alias:
        alias = alias.split('-Run')[0]
    return alias

@dataclass
class FalconContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    # match FalconConfig.hash_dataset() functionality to get a unique ID for a dataset that is provided by evaluator
    # This method returns unique per SESSION, doesn't/shouldn't distinguish train/eval split
    @classmethod
    def explicit_session_reduction(cls, alias: str) -> str:
        if 'FALCONH1' in alias:
            # breakpoint()
            alias = alias.replace('-', '_')
            pieces = alias.split('_')
            if '1925' in alias: # dandi formatted - note this alias doesn't correspond to EvalAI aliases
                s_key = pieces[-1].split('T')[0][4:]
                s_index = ['0101', '0108', '0113', '0115', '0119', '0120', '0126', '0127', '0129', '0202', '0203', '0206', '0209'].index(s_key)
                return f'S{s_index}'
            for piece in pieces:
                if piece[0] == 'S':
                    return piece
            raise ValueError(f"Could not find session in {alias}.")
        if 'FALCONM1' in alias:
            if 'behavior+ecephys' in alias:
                return alias.split('_')[-2].split('-')[-1]
            elif 'val' in alias: # val, ExperimentalTask.falcon_m1-FALCONM1-falcon_FALCONM1-L_20120926_held_in_eval
                return alias.split('FALCONM1-L_')[1].split('_')[0]
        if 'FALCONH2' in alias:
            if 'sub-T5' in alias:
                return alias.split('-')[-1]
            # 'falcon_FALCONH2-T5_2022.05.18_held_in_eval' -> T5_2022.05.18_held_in_eval
            return alias.split('-')[-1]
        if 'FALCONM2' in alias:
            if 'ses-' in alias:
                return alias.split('ses-')[1].split('-Run')[0]
                # return alias.split('ses-')[1].split('_')[0]
            elif 'val' in alias: # minival or eval, formatted differently
                # falcon_minival_FALCONM2-sub-MonkeyNRun1_20201019_held_in_minival
                # run_num = alias.split('Run')[-1].split('_')[0]
                date_str = alias.split('_')[-4]
                return f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}' # -Run{run_num}'
        raise NotImplementedError(f"Session reduction not implemented for {alias}")

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["M1"], suffix='', alias_prefix=''):
        root = Path(root)
        if not root.exists():
            logger.warning(f"Datapath folder {root} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # path = ..../h1/
            if task == ExperimentalTask.falcon_h1:
                subject = 'h1'
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                # breakpoint()
                # Do not differentiate phase split OR set in session for easy transfer - phase split follows set annotation
                # ! TODO nope, we can't afford this - we get registry collisions
                # pieces = path.stem.split('_')
                # pre_set_pieces = pieces[:pieces.index('set')]
                # stem = '_'.join(pre_set_pieces)
                # print(path)
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{subject.name.value}-{path.stem}", # Be careful to not overwrite
                    datapath=path,
                )
            elif task == ExperimentalTask.falcon_h2:
                subject = 'h2'
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{subject.name.value}-{path.stem}",
                    datapath=path,
                )
            elif task == ExperimentalTask.falcon_m1:
                # sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb
                subject = "m1"
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{alias_prefix}{subject.name.value}-{path.stem}",
                    datapath=path,
                )
            # elif task == ExperimentalTask.mock_half_falcon_m1:
            #     subject = "m1"
            #     subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
            #     return FalconContextInfo(
            #         subject=subject,
            #         task=task,
            #         _arrays=arrays,
            #         alias=f"mock_{mock_half}_falcon_{subject.name.value}-{path.stem}",
            #         datapath=path,
            #     )
            elif task == ExperimentalTask.falcon_m2:
                subject = 'm2'
                subject = SubjectArrayRegistry.query_by_subject(f'falcon_{subject}')
                return FalconContextInfo(
                    subject=subject,
                    task=task,
                    _arrays=arrays,
                    alias=f"falcon_{alias_prefix}{subject.name.value}-{path.stem}",
                    datapath=path,
                )
                # sub-MonkeyN-held-in-calib_ses-2020-10-19-Run1_behavior+ecephys.nwb
                session_hash = stem.split('ses-')[1].split('_')[0]
                return f"falcon_{subject.value}-{session_hash}"
        if suffix:
            infos = map(make_info, root.glob(f"*{suffix}*.nwb"))
        else:
            infos = map(make_info, root.glob("*.nwb"))
        return list(filter(lambda x: x is not None, infos))

@dataclass
class DANDIContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        path = Path(root)
        if not path.exists():
            logger.warning(f"Datapath folder {path} does not exist. Skipping.")
            return []
        def make_info(path: Path): # path e.g. DANDI_ID/sub-C/sub-C_ses-CO-20150629_behavior+ecephys.nwb
            subject = f"{task.value}_{'_'.join(path.parts[-2].split('-')[1:]).lower()}"
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return DANDIContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{task.value}_{subject.name.value}_{path.stem.split('_')[-2]}",
                datapath=path,
            )
        infos = map(make_info, path.glob("*.nwb"))
        return filter(lambda x: x is not None, infos)

@dataclass
class NHContextInfo(ContextInfo):
    # HatsopoulosLab

    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        path = Path(root)
        if not path.exists():
            logger.warning(f"Datapath folder {path} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # If located in CO folder, special, for analysis
            subject_kw = root.split('data/hatlab/')[1].split('/')[0].lower()
            subject = SubjectArrayRegistry.query_by_subject('hat_generic')
            return NHContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{task.value}_{subject_kw}_{path.stem}", # i.e. hatsopoulos_L2015...
                datapath=path,
            )
        def make_info_from_exp_folder(path: Path):
            subject = SubjectArrayRegistry.query_by_subject('hat_generic')
            return NHContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{task.value}_{path.parent.parts[-2]}_{path.parent.parts[-1]}",
                datapath=path,
            )
        if 'CO/Velma' in str(path.parent):
            infos = map(make_info_from_exp_folder, path.glob("*.mat"))
        else:
            infos = map(make_info, path.glob("*.nev"))
        return filter(lambda x: x is not None, infos)

    @classmethod
    def build_from_nested_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        r"""
            Recurses one deep - checks for presence of NSX, if so, assume "main" directory, else assume year.
            Also - there may be multiple NSX, return one Ctx per NSX.
        """
        path = Path(root)
        for subdir in path.glob("*"):
            if path.is_dir():
                infos = cls.build_from_dir(str(subdir), task, arrays)
                yield from filter(lambda x: x is not None, infos)

@dataclass
class LimbLabContextInfo(ContextInfo):
    # Limblab (Lee Miller)

    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        path = Path(root)
        if not path.exists():
            logger.warning(f"Datapath folder {path} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # should be data / limblab /subject / timestamp / path
            subject_kw = root.split('data/limblab/')[1].split('/')[0].lower()
            session_timestamp = root.split('data/limblab/')[1].split('/')[1]
            subject = SubjectArrayRegistry.query_by_subject('limblab_generic')
            return LimbLabContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{task.value}_{subject_kw}_{session_timestamp}_{path.stem}",
                datapath=path,
            )
        infos = map(make_info, path.glob("*.nev"))
        return filter(lambda x: x is not None, infos)

    @classmethod
    def build_from_nested_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        r"""
            Recurses one deep - checks for presence of NSX, if so, assume "main" directory, else assume year.
            Also - there may be multiple NSX, return one Ctx per NSX.
        """
        path = Path(root)
        for subdir in path.glob("*"):
            if path.is_dir():
                infos = cls.build_from_dir(str(subdir), task, arrays)
                yield from filter(lambda x: x is not None, infos)

@dataclass
class DeoContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        path = Path(root)
        if not path.exists():
            logger.warning(f"Datapath folder {path} does not exist. Skipping.")
            return []
        def make_info(path: Path):
            # should be data
            subject, timestamp = path.name.split('_', 1)
            subject = SubjectArrayRegistry.query_by_subject(subject)
            return DeoContextInfo(
                subject=subject,
                task=task,
                _arrays=arrays,
                alias=f"{task.value}_{subject.name.value}_{path.stem}",
                datapath=path,
            )
        infos = map(make_info, path.glob("*.mat"))
        return filter(lambda x: x is not None, infos)

@dataclass
class MooreContextInfo(ContextInfo):
    def _id(self):
        return self.alias

    @classmethod
    def build_from_dir(cls, root: str, task: ExperimentalTask, arrays=["main"]):
        path = Path(root)
        if not path.exists():
            logger.warning(f"Datapath folder {path} does not exist. Skipping.")
            return []