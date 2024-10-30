
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from context_general_bci.config import REACH_DEFAULT_KIN_LABELS, REACH_DEFAULT_3D_KIN_LABELS
from context_general_bci.tasks.myow_co import DYER_DEFAULT_KIN_LABELS
from context_general_bci.tasks.miller import MILLER_LABELS

MARKER_SIZE = 120

global_palette = sns.color_palette('colorblind', n_colors=11)
# global_palette = sns.color_palette('colorblind', n_colors=7)
colormap = {
    'scratch': global_palette[0],
    'scratch_transfer': global_palette[0],
    'NDT3': global_palette[0],
    'NDT3 Expert': global_palette[0],
    'NDT3 mse': global_palette[0],
    'NDT2 Expert': global_palette[1],
    'base_45m_200h': global_palette[2],
    'base_45m_200h_mse': global_palette[2],
    'base_45m_200h_transfer': global_palette[2],
    'base_45m_200h_smth': global_palette[2],
    'base_45m_200h_linear': global_palette[2],
    'big_350m_200h': global_palette[2],
    'huge_700m_200h': global_palette[2],
    'base_45m_1kh': global_palette[4],
    'base_45m_1kh_mse': global_palette[4],
    'base_45m_1kh_smth': global_palette[4],
    'base_45m_1kh_linear': global_palette[4],
    'big_350m_1kh_smth': global_palette[4],
    'big_350m_1kh_linear': global_palette[4],
    'base_45m_1kh_human': global_palette[5],
    'base_45m_1kh_human_mse': global_palette[5],
    'base_45m_1kh_human_smth': global_palette[5],
    'base_45m_1kh_human_linear': global_palette[5],
    'base_45m_2kh': global_palette[3],
    'base_45m_2kh_smth': global_palette[3],
    'base_45m_2kh_linear': global_palette[3],
    'base_45m_2kh_mse': global_palette[3],
    'big_350m_2kh_smth': global_palette[3],
    'big_350m_2kh': global_palette[3],
    'big_350m_2kh_transfer': global_palette[3],
    'huge_700m_2kh': global_palette[3],
    '200h_ablate_mask': 'red',
    # AHHHH
    'base_45m_min': global_palette[6],
    'base_45m_25h': global_palette[7],
    'base_45m_70h': global_palette[8],
    'base_45m_1kh_breadth': global_palette[9],
    'big_350m_2500h': global_palette[10],
    'ole': 'k',
    'wf': 'k',
    'ridge': 'k',
}

pt_volume_labels = {
    'cursor': ['2.5 min', '5 min', '10 min'],
    'cursor_new': ['2.5 min', '5 min', '10 min'],
    'falcon_h1': ['15 min', '30 min', '1 hr'],
    'falcon_m1': ['1h', '2h', '4 hr'],
    'falcon_m2': ['11 min', '22 min', '44 min'],
    'rtt': ['3h', '6h', '12 hr'],
    'grasp_h': ['15 min', '30 min', '1 hr'],
    'grasp_new': ['10 min', '20 min', '40 min'],
    'grasp_v3': ['3 min', '6 min', '12 min'],
    'rtt_s1': ['7 min', '14 min', '28 min'],
    'cst': ['10 min', '21 min', '42 min'],
    'neural_cst': ['10 min', '21 min', '42 min'],
    'eye': ['2.5h', '5h', '10 hr'],
    'bimanual': ['42 min'], # 2545, 9 x 4 minutes per.
}

heldin_tune_volume_labels = {
    'falcon_h1': ('8 min', 6),
    'falcon_m1': ('1 hr', 4),
    'falcon_m2': ('10 min', 4),
}

# Data available in evaluation sessions
# r'64s$\times$4' for raw latex string
tune_volume_labels = {
    'cursor': ('60 s', 11),
    'cursor_new': ('60 s', 11),
    'falcon_h1': ('80 s', 7),
    'falcon_m1': ('2 min', 4),
    'falcon_m2': ('64 s', 4),  # Assuming None for missing numerical value
    'rtt': ('60 s', 3),  # Assuming None for missing numerical value
    'grasp_h': ('10 min', 6),  # Assuming None for missing numerical value
    'grasp_new': ('10 min', 3),  # Assuming None for missing numerical value
    'grasp_v3': ('4 min', 3),
    'cst': ('60 s', 39),
    'neural_cst': ('60 s', 39),
    'rtt_s1': ('4 min', 7),
    'eye': ("40 min", 1),
    'bimanual': ('5 min', 9),
}

def variant_volume_map(variant_stem):
    if 'Expert' in variant_stem:
        return 0
    elif '200h' in variant_stem:
        return 200
    elif '1kh' in variant_stem:
        return 1000
    elif '2kh' in variant_stem:
        return 2000
    elif '25h' in variant_stem:
        return 25
    elif '70h' in variant_stem:
        return 70
    elif 'min' in variant_stem:
        return 1.5
    else:
        return 0

UNIQUE_PT_VOLUMES = [1.5, 25, 70, 200, 1000, 2000]

# Normalize UNIQUE_PT_VOLUMES to 0-1 for a logscale colormap
log_unique_pt_volumes = np.log10(UNIQUE_PT_VOLUMES)
normalized_unique_pt_volumes = (log_unique_pt_volumes - log_unique_pt_volumes.min()) / (log_unique_pt_volumes.max() - log_unique_pt_volumes.min())
cont_size_palette = sns.color_palette("coolwarm", as_cmap=True)
SIZE_PALETTE = {}
for i, vol in enumerate(UNIQUE_PT_VOLUMES):
    SIZE_PALETTE[vol] = cont_size_palette(normalized_unique_pt_volumes[i])



# Old NDT2 styleguide, I think

STYLEGUIDE = {
    "palette": sns.color_palette('colorblind', 5),
    "hue_order": ['single', 'session', 'subject', 'task'],
    "markers": {
        'single': 'o',
        'session': 'D',
        'subject': 's',
        'task': 'X',
    }
}

CAMERA_LABEL = { # from model internal label to camera-ready label
    "x": "Vel X",
    "y": "Vel Y",
    "z": "Vel Z",
    "EMG_FCU": "FCU",
    "EMG_ECRl": "ECRl",
    "EMG_FDP": "FDP",
    "EMG_FCR": "FCR",
    "EMG_ECRb": "ECRb",
    "EMG_EDCr": "EDCr",
    'f': 'Force',
}

DIMS = {
    'gallego': REACH_DEFAULT_KIN_LABELS,
    'dyer': DYER_DEFAULT_KIN_LABELS,
    'miller': MILLER_LABELS,
    'churchland_misc': REACH_DEFAULT_3D_KIN_LABELS,
    'churchland_maze': REACH_DEFAULT_KIN_LABELS,
    'delay': REACH_DEFAULT_3D_KIN_LABELS,
    'odoherty': REACH_DEFAULT_KIN_LABELS,
}

def prep_plt(ax=None, **kwargs) -> plt.Axes:
    if isinstance(ax, np.ndarray):
        for _ax in ax.ravel():
            _prep_plt(_ax, **kwargs)
    else:
        ax = _prep_plt(ax, **kwargs)
    return ax

def _prep_plt(ax=None, spine_alpha=0.3, size="small", big=False):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    LARGE_SIZE = 15
    if big:
        size = "large"
    if size == "medium":
        SMALL_SIZE = 12
        MEDIUM_SIZE = 16
        LARGE_SIZE = 20
    if size == "large":
        SMALL_SIZE = 18
        # SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        # MEDIUM_SIZE = 24
        LARGE_SIZE = 26
        # LARGE_SIZE = 28
    if size == "huge":
        SMALL_SIZE = 22
        MEDIUM_SIZE = 26
        LARGE_SIZE = 30
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.style.use('seaborn-v0_8-muted')
    if ax is None:
        plt.figure(figsize=(6,4))
        ax = plt.gca()
    ax.spines['bottom'].set_alpha(spine_alpha)
    ax.spines['left'].set_alpha(spine_alpha)
    # ax.spines['top'].set_alpha(spine_alpha)
    ax.spines['top'].set_alpha(0)
    # ax.spines['right'].set_alpha(spine_alpha)
    ax.spines['right'].set_alpha(0)
    ax.grid(alpha=0.25)
    return ax

def data_label_to_target(data_label: str):
    if data_label == 'dyer':
        target = ['dyer_co_chewie_2']
    elif data_label == 'gallego':
        target = ['gallego_co_.*']
    elif data_label == 'rouse':
        target = ['rouse_.*']
    elif data_label == 'churchland':
        target = ['churchland_maze_jenkins.*']
    elif data_label == 'loco':
        target = [
            'odoherty_rtt-Loco-20170215_02',
            'odoherty_rtt-Loco-20170216_02',
            'odoherty_rtt-Loco-20170217_02',
        ]
    elif data_label == 'indy': # EVAL SET
        target = [
            'odoherty_rtt-Indy-20160407_02', # First indy session
            'odoherty_rtt-Indy-20160627_01', # Original
            'odoherty_rtt-Indy-20161005_06',
            'odoherty_rtt-Indy-20161026_03',
            'odoherty_rtt-Indy-20170131_02'
        ]
    elif data_label == 'miller':
        target = [
            'miller_Jango-Jango_20150730_001',
            'miller_Jango-Jango_20150731_001',
            'miller_Jango-Jango_20150801_001',
            'miller_Jango-Jango_20150805_001'
        ]
    elif data_label == 'eval':
        target = [
            'dyer_co_chewie_2',
            'odoherty_rtt-Indy-20160407_02', # First indy session
            'odoherty_rtt-Indy-20160627_01', # Original
            'odoherty_rtt-Indy-20161005_06',
            'odoherty_rtt-Indy-20161026_03',
            'odoherty_rtt-Indy-20170131_02',
            'miller_Jango-Jango_20150730_001',
            'miller_Jango-Jango_20150731_001',
            'miller_Jango-Jango_20150801_001',
            'miller_Jango-Jango_20150805_001'
        ]
    elif data_label == 'robust':
        target = [
            'odoherty_rtt-Indy-20160627_01'
        ]
    elif data_label == 'p4_grasp':
        # manual sampling https://pitt-my.sharepoint.com/:x:/r/personal/ghb14_pitt_edu/_layouts/15/Doc.aspx?sourcedoc=%7BEFCBDF63-B37B-4C60-A578-0A51AEE4157B%7D&file=U01%20Testing%20Dates.xlsx&action=default&mobileredirect=true
        target = [
            'pitt_broad_pitt_co_P4Lab_9_.*',
            # 'pitt_broad_pitt_co_P4Lab_10_.*',
            # 'pitt_broad_pitt_co_P4Lab_13_.*',
            # 'pitt_broad_pitt_co_P4Lab_14_.*',
            # 'pitt_broad_pitt_co_P4Lab_36_.*',
        ]
    elif data_label == 'p4':
        target = [
            'pitt_broad_pitt_co_P4.*',
        ]
    elif data_label == 'indy_miller':
        target = [
            'odoherty_rtt-Indy-20160407_02', # First indy session
            'odoherty_rtt-Indy-20160627_01', # Original
            'odoherty_rtt-Indy-20161005_06',
            'odoherty_rtt-Indy-20161026_03',
            'odoherty_rtt-Indy-20170131_02',
            "miller_Jango-Jango_20150730_001",
            "miller_Jango-Jango_20150731_001",
            "miller_Jango-Jango_20150801_001",
            "miller_Jango-Jango_20150805_001",
        ]
    else:
        raise ValueError(f"Unknown data label: {data_label}")
    return target
