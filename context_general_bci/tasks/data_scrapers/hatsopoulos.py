from typing import List
import os
from pathlib import Path
import re
import shutil
import subprocess
# https://chat.openai.com/share/5d54e5ce-94db-4955-aa59-dd4865dde6db
LOCAL_MTPT = '/mnt/x'
INTERMEDIATE_PT = '/home/joy47/hatsopoulos_data/'
# INTERMEDIATE_PT = '/home/joel/hatsopoulos_data/'
REMOTE_MTPT = 'crc:projects/context_general_bci/data/hatsopoulos/'
# Scrape data from hatsopoulos share drive.
r"""
    1. We could just hunt for every NEV and pull it down, but NEV is unstructured and this will maybe 10% efficient...? - we don't have _that_ much space on CRC.
    - But this leaves out certain .mats - certainly not every .mat is tractable.


    # Recommended from Nicho: Use Collaboratorsdata
    - JY sees also a raw data repo, which appears to store mostly recent monkeys.
    - Together, we still see about 8-10 monkeys - a bit lower than expected but plenty to work with... (any more?)

    - Of all our data, Nicho's is the most heterogeneous and inconsistently structured. We'll end up pulling in a lot of data with diff formats to process. Let's try to keep this tractable.
    - Heterogeneity true both in CollaboratorsData as well as all_raw_datafiles.
    - Reasonable effort applied to make sure we're not pulling duplicate files
    - For tractability, we ONLY INCLUDE NEURAL DATA. Parsing various covariates formats too expensive for JY right now (may want to go back and select a few)

    - Neural formats either .mats or .nevs - prefer .mats when available since data is smaller.

    Notes on formatting
    General:
    - Data is highly heterogeneous but low session count, even within task/monkey pair. Prefer .mats over .nevs, to reduce datafile size.
    - Interesting files pulled manually based on JY's inspection of dir contents. Mainly aiming for raw neural data.

    - Wheel task
        - 2 monkeys, MI in .NSx/.NEV files, long tail of other available files
        - IIRC NEV is all I want, NS2 is 1kHz (LFP)
    - Target-jump (eblow movement only)
        - 3 monkeys, .mats ; also has psth/lfp files (ending with _psth.mat)
        - seems robust to check: len(stem.split('_')) = 2.
        - Multiarea split across days, either merge or treat separately.
    - Sleep: Has some all sleep, some day / night recordings. Cool!
        - Challenge for sleep files - different areas appear noted in different files, sometimes with different handles. Not scalable to treat together - JY advocates one single catch-all array register for Nicho's monkeys. (Not a huge loss)
    - RTP
        - Some RTP dirs have processed data forms; we want raw. Heuristic for filestem is candidate.lower().split('_')[0] == session_id.lower().split('_')[0] and is shortest stem (no postprocess suffix). But let's hardcode a few to avoid heuristics.
        - List hardcodes before general queries, and exclude dirs already proced by hardcode
    - ReachGrasp

    Notes for Nicho:
    - I'm pulling Collaborator data and some raw data under \Data\all_raw_datafiles_7\. all_raw_datafiles doesn't seem very complete - is there another repo to consider?
    - I'm pulling a mix of .mats and .nevs - skipping kin for now
    - Hermes has many sessiosn with zipped .c3d files. Are those important? They might be a good candidate for large, consistent source of covariate if that's what they are.


    Questions for Nicho:
    - M1 or MI as area suffix in some paths? Why?
    - RTP protocol: What does this stand for?
    - In raw data files, often CTD
    - In various dirs, sometimes NEVs are tiny but substantial # of other files indicates it's a real experimental day. Are these stim days where NEV is disabled?

    Challenges:
    - Full data pull is huge (O(TB)) - need to pull carefully...

    General commands:
    - If extension is not specified, check for .mat and then .nev.
    - For each level past subject directory, if a path doesn't exist, fuzzy search for a match with str dist less than 2, and replace that leve with the match. Log this process carefully. Pause for interaction if either multiple matches or no matches show up.
    - Print warnings when either multiple data files or no data files show up for an input
    - Some raw windows paths are provided - convert to the appropriate linux path object
    - All greps with tags should be case-insensitive
"""
INFORMAL_REGEX_TO_REGEX = {
    'YYYYMMDD': r'(\d{4}\d{2}\d{2})',
    'YYMMDD': r'(\d{2}\d{2}\d{2})',
    'YYYY-MM-DD': r'(\d{4})-(\d{2})-(\d{2})',
    'YYMonDD': r'(\d{2}(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2})',
    'tag': r'(.*)',
    'BrainArea': r'(MI|M1|PMd|PMv)'
}
# If not in this list, assume a generic regex that matches a comma separated list of strs

STOPWORDS = ["Vicon", "Kin", "lfp"] # Ignoring this data for now
STOPWORDS = [ s.lower() for s in STOPWORDS ]
root = Path("Collaboratorsdata")
exp_root = root # Top level, no op for later
path_collapse = {
    'Target-jump/1D Target Jump (elbow movement only)': 'target_jump',
    # 'CenterOut': 'center_out',
    # 'ReachGrasp': 'reach_grasp',
}

exp_roots_of_interest = [
    # These are generally _files_
    Path('Target-jump') / '1D Target Jump (elbow movement only)' / "Boo" / '*{YYMonDD}_{BrainArea}.mat',
    Path('Target-jump') / '1D Target Jump (elbow movement only)' / "Coco" / '*{YYMonDD}_{BrainArea}.mat',
    Path('Target-jump') / '1D Target Jump (elbow movement only)' / "Mack" / "*{YYMonDD}_{BrainArea}.mat",

    # these are of the format "CenterOut/Boo/{session_tag}_MI"
    # these are of the format "CenterOut/Boo/{session_tag}_Vicon"
    # e.g. the folders specify subtype of data
    # hard to think of simple regex logic - we block bad suffixes with stop words

    Path("CenterOut") / "Boo" / "*" / "*.mat", # should be single match
    Path("CenterOut") / "Coco" / "*" / "*.mat",
    Path("CenterOut") / "Mack" / "*" / "*.mat",
    Path("CenterOut") / "Niki" / "*" / "*.mat",
    Path("CenterOut") / "Raju" / "*" / "*.mat",
    Path("CenterOut") / "Roxie" / "*" / "*.mat",
    Path("CenterOut") / "Velma" / "*" / "*.mat",

    Path("CenterOut") / "Rockstar" / "rs1050225" / "rs1050225_clean_SNRgt4.mat",
    Path("CenterOut") / "Rockstar" / "rs1051013_both ctr-out_and_RTP" / "rs1051013_clean_SNRgt4.mat",

    # Removed to avoid duplicating with Raw poll, which is less heterogeneous.
    # Lester CO - 2015/09/30 - 2017/03/22
    # Path("CenterOut" / "Lester" / "Ls{yymmdd}_woEMG" / "Ls{yymmdd}LAm_munits.mat", # Low confidence, data file seems small (2Mb vs .5G NEV). Parent dir should have ~3 matches.
    # Path("CenterOut" / "Lester" / "Ls{yymmdd}_woEMG" / "Ls{yymmdd}M1Lemg.nev",  # Hm. This seems too long tail for now.
    # Path("CenterOut" / "Lester" / "{yymmdd}_NoEMG" / "*.nev" # one match


    # Path("CenterOut" / "Breaux" / "Bx{YYMMDD}_withEMG" / "{YYMMDD}*_singleunits_VPsorted.mat", # possibly multiple match or no match # Long tail

    Path("CenterOut") / "Breaux" / "Bx190228_withEMG" / "Bx190228M1m_RTP_units_sorted_CS.mat",
    Path("CenterOut") / "Breaux" / "Bx190228_withEMG" / "Bx190228M1l_RTP_units_sorted_CS.mat",
    Path("CenterOut") / "Breaux" / "Bx190228_withEMG" / "Bx190228M1m_CO_units_sorted_CS.mat",
    Path("CenterOut") / "Breaux" / "Bx190228_withEMG" / "Bx190228M1l_CO_units_sorted_CS.mat",

    # Path("Sleep" / "{subject}" / "Matlab files" / "{tag}N" / "{tag}N{nsp_no}_IDs", # suffix: IDs or spikes, both interesting? There are also NEV / NS2 files. PMc and PMd. Whole stem might be usable as an ID.
    # Path("Sleep" / "Velma" / "*" / "*.nev", # Boo also available but format different
    # Path("Sleep" / "Boo" / "*.nev", # Also has .mats but prefer .nev for uniformity with above (no need for extra parser)
    # ! Be careful about multiple arrays for a single "hash". Array ID-ing is pretty variable, so we might not be able to avoid dup here.
    Path("Sleep") / "*" / "*" / "*.nev", # Boo also available but format different

    # Path("RTP" / "{Raju,Rockstar}" / "{session_id}" / "*.mat" # session ID highly varied across monkeys. Mack blacklisted - has exploded spikes across mats, IDK.
    Path("RTP") / "Raju" / "r1031206_PMd_MI" / "R1031206_PMdab_M1bc002_cleanspikes.mat",
    Path("RTP") / "Raju" / "r1031126_M1all_PMdc" / "r1031126_M1all_PMdc_SNR_gt4.mat",
    Path("RTP") / "Rockstar" / "rs1041130_MI" / "rs1041130_MIall_clean_spikes_SNRgt4.mat",
    # Path("RTP" / "Rockstar" / "rs1050211_MI" / "units.mat", # Long tail

    # Reach Grasp
    Path("ReachGrasp") / "Athena" / "A5212008" / "ALL DATA ALIGNED" / "Athena_05_21_2008DATA.mat", # note kin is available but ignore for now
    Path("ReachGrasp") / "Athena" / "A5222008" / "ALL DATA ALIGNED" / "Athena_05_22_2008DATA.mat",
    Path("ReachGrasp") / "Oreo" /  "05022008" / "ALL DATA ALIGNED" / "Oreo_MI_PMv_05_01_08DATA.mat",
    Path("ReachGrasp") / "Oreo" /  "05132008" / "ALL DATA ALIGNED" / "Oreo_05_13_08_MI_PMvDATA.mat",
    Path("ReachGrasp") / "Oreo" /  "05132008" / "ALL DATA ALIGNED" / "Oreo_05_13_08_MI_PMvDATA.mat",

    Path("ReachGrasp") / "Oreo" /  "20080428" / "{BrainArea}" / "*.nev",
    Path("ReachGrasp") / "Oreo" /  "5.07.2008" / "unsorted" / "*.nev", # multi-match OK
    Path("ReachGrasp") / "Jaco" / "{YYYYMMDD}" / "{BrainArea}" / "*.nev", # should be single match, e.g. J121031_PMv001.nev.
    # At {PMd,PMv} level - some don't have data, some have ViconData and Kin - skip all these cases. Just keep if {PMd,PMv,M1}
    # At final level - we may have LFP/.mats -- ignore. If multiple NEV, take the one whose stem string includes brain area specified in parent dir. If no match, continue.

    # Lester ReachGrasp - ~2011/12/29 to 2013/09/18
    Path("ReachGrasp") / "Lester" / "{YYYYMMDD}" / "{BrainArea}" / "*.nev", # same logic as Jaco

    # Marmosets
    Path("marmosets") / "TY20170512" / "TY20170512_001_sortedSpikes.mat",
    Path("marmosets") / "TY20210328_1909_inHammock_earlyEvening002_processed.mat",
    Path("marmosets") / "TY20210329_2008_inHammock_earlyEvening001_processed.mat",
    # Wheel task
    Path("Wheel") / "Athena" / "*" / "{BrainArea}" / "*.nev" # <= 1 match per subdir ok, should have at least one match for tag
]

r"""
    Notes on raw data
    - Some sessions have special annotations at the datetime dir level, e.g. may be non-monkey system tests.
    - In each session, we're looking for just raw recordings for uniform format. Some have processed subdirs or zips - ignore.
    - However, there are also many days with just NSx and NEV files. (here, but also in collaborator data) -- what are these days? Am I missing data in NSx?
        - e.g. "X:\Data\all_raw_datafiles_7\Hermes\210322\Hm210322M1lPM.nev"

    Commands:
    - Skip excessively large files (>2G.) and excessively small files (<2Mb, NEV likely disabled)
    - In case of several matches with identical sizes, keep the one that has stem with the closest fuzzy match to datetime str.
"""
raw_root = Path('Data') / 'all_raw_datafiles_7'
# This is kept in a format preferred to be human readable, but should be compiled to proper regex.
raw_roots_of_interest = [
    Path('Jim') / '{YYYYMMDD}' / '*{YYYYMMDD}*.nev', # expecting <= 10 nevs per session; discard files < 2Mb (NEVs < 2Mb likely non-neural). Ensure NEV is somewhere between 1Mb and 2G, else warn.
    Path('Hermes') / '{YYMMDD}' / '*{YYMMDD}*.nev',

    Path('Theseus') / '2021' / '{YYMMDD}' / '*.nev', # Hm.. some of these zips have c3d, what is that?
    Path('Theseus') / '2022' / '{YYMMDD}' / '*.nev', # There are some nested dirs with resting data. TODO Step in into them and checking filestems (I think the filestem is the same) e.g. candidat_dirs = self + dirs in self. Check all nevs, then cull by unique filesize. Meh... low priority, we lose metadata about what data it is.

    # Careful with these for dup-ing with above
    # 2015 - 2017
    Path('Lester') / '{YYYYMMDD}' / "*.nev", # appears to have EMG. Several Gb a file. May not be worth it (e.g. 100G for this data alone.). Possibly just skip overly large files.
    # raw_root / 'Lester' / '2017' / '{YYMMDD}' / "*.nev", # Long tail format, 3 sessions

    # This 2015 data has ICMS (in M1?)!!! Oh well..
    Path('Lester') / '2015' / '{YYYYMMDD}' / "*.nev",
    Path('Lester') / '2015' / '{YYMMDD}' / "*.nev",
    Path('Lester') / '2016' / '{YYYY-MM-DD}' / "*.nev",
    Path('Lester') / '2016' / '{YYMMDD}' / "*.nev",
    Path('Lester') / '2016' / '{YYYYMMDD}' / "*.nev",

    # 2017-2021. 20/21 long tail.
    # * Hm. This format is not ruly. There's a lot of files in these backups.
    Path('Breaux') / '2017' / '{YYMMDD}' / '*.nev', # A few sessions in here with nontrivial NEVs; parse by filesize rules
    # raw_root / 'Breaux' / '2018', # Breaux appears to be doing ICMS from 2018 onward
    # raw_root / 'Breaux' / '2019', # no useful NEVs on many spot checks
]

def find_placeholders(s):
    r"""
        # Examples
        template_str = "This is a template string with ${placeholder} and {another_placeholder}"
        format_str = "This string uses {placeholder} for formatting and {another_one}"
        regular_str = "This is a regular string without placeholders."

        print(find_placeholders(template_str))  # ['placeholder', 'another_placeholder']
        print(find_placeholders(format_str))    # ['placeholder', 'another_one']
        print(find_placeholders(regular_str))   # []
    """
    # Matches placeholders within {} for str.format() and ${} for template strings
    template_pattern = re.compile(r'\$\{([^}]*)\}|\{([^}]*)\}')
    matches = template_pattern.findall(s)

    # Each match is a tuple, where only one element is the actual placeholder name
    # Flatten the list and filter out empty strings
    placeholders = [placeholder for match in matches for placeholder in match if placeholder]
    return placeholders

def compile_path_to_regex(path_of_interest: Path):
    query = str(path_of_interest)
    placeholders = find_placeholders(query)
    # First do a pass for backreferences
    seen_ctx = []
    # Escape random critical characters
    query = re.escape(query).replace(r'\*', '.*') # Restore wildcard syntax
    for p in placeholders: # These replacements should only replace one at a time, in a forward scan
        if p in seen_ctx:
            query = query.replace("\\{" + p + "\\}", f"\\{seen_ctx.index(p) + 1}", 1) # reduce to backreferences
        else:
            query = query.replace("\\{" + p + "\\}", INFORMAL_REGEX_TO_REGEX[p], 1)
            seen_ctx.append(p)
    # Compile to true regex
    # Implicit leading wildcard
    compiled = re.compile('.*' + query)
    return compiled

COMPILED_COLLAB = [compile_path_to_regex(p) for p in exp_roots_of_interest]
COMPILED_RAWS = [compile_path_to_regex(raw_root / p) for p in raw_roots_of_interest]
# COMPILED_ALL = [*COMPILED_COLLAB, *COMPILED_RAWS]

def is_valid_file(file_path):
    file = Path(file_path)
    if not file.is_file():
        return False
    if not file.suffix in ['.mat', '.nev']:
        return False
    if any(i in str(file_path).lower() for i in STOPWORDS):
        return False
    size = file.stat().st_size
    return 1e6 <= size <= 2e9

# ! This is too slow in python. Go through shell instead. Main issue is that it doesn't warn about expectations... but that's maybe ok.
def find_matching_files(root: Path, patterns: List[re.Pattern]):
    matches = []
    for dirpath, dirnames, fns in os.walk(root):
        print(dirpath, len(matches))
        # Not the most efficient but...
        # No dirname will match - we're checking only for files
        dir_fns = [os.path.join(dirpath, fn) for fn in fns]
        dir_fns = [fn for fn in dir_fns if is_valid_file(fn)]
        print(dir_fns)
        if dir_fns:
            for pattern in patterns:
                cur_match = list(filter(pattern.match, dir_fns))
                matches.extend(cur_match)
    return matches
    # Ideally we can zip everything... but it would be huge...

def find_matching_files_with_shell(root: Path, pattern: re.Pattern, min_size_mb=1, max_size_gb=3) -> List[str]:
    print(f"Try match: {root}, {pattern}")
    try:
        # Construct the find command
        min_size_param = f"+{min_size_mb}M"
        max_size_param = f"-{max_size_gb}G"

        find_command = ["find", str(root), "-type", "f", "-size", min_size_param, "-size", max_size_param]
        find_command = ["find", str(root), "-type", "f"] # , "-size", min_size_param, "-size", max_size_param]

        # Execute the find command
        find_process = subprocess.Popen(find_command, stdout=subprocess.PIPE)
        # Use grep to filter the find results with the regex pattern. -i for case insensitivity
        grep_process = subprocess.Popen(["grep", "-iP", pattern.pattern], stdin=find_process.stdout, stdout=subprocess.PIPE, text=True)
        # Close find_process' stdout to allow it to receive a SIGPIPE if grep_process exits
        find_process.stdout.close()
        # Get the output from grep
        output, errors = grep_process.communicate()
        out = output.splitlines()  # Return a list of matching files
        out = [o for o in out if not any(stop in o for stop in STOPWORDS)]
        return out
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# sample_path = Path(LOCAL_MTPT) / raw_root / 'Jim' #  / "20200925" # /J20200925001.nev"
# print(find_matching_files(sample_path, [COMPILED_RAWS[0]]))
# test = find_matching_files_with_shell(sample_path, COMPILED_RAWS[0])
# print(f"Found: {len(test)}")
# sample_path = Path(LOCAL_MTPT) / raw_root / 'Jim' #  / "20200925" # /J20200925001.nev"
def get_and_shuttle_matching_paths(
    paths: List[Path],
    patterns: List[re.Pattern],
    path_root=Path(LOCAL_MTPT),
    shuttle_target=Path(INTERMEDIATE_PT),
):
    for search_path, pattern in zip(paths, patterns):
        pieces = search_path.parts
        search_root = None
        for i, piece in enumerate(pieces):
            if find_placeholders(piece) or "*" in piece:
                search_root = Path(*pieces[:i])
                break
        if search_root is None:
            # true path, just return
            search_results = [path_root / search_path]
        else:
            # search_results = find_matching_files(path_root / search_root, [pattern])
            search_results = find_matching_files_with_shell(path_root / search_root, pattern)
        print(f'Found {len(search_results)} for {search_path}\n')#  | {pattern.pattern}')

        # shuttle
        search_relative = [Path(sr).relative_to(path_root) for sr in search_results]
        for sr in search_relative:
            sr_tgt = sr
            for shorten in path_collapse:
                if shorten in str(sr):
                    sr_tgt = str(sr).replace(shorten, path_collapse[shorten])
            target_path = shuttle_target / sr_tgt
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path_root / sr, target_path)

# get_and_shuttle_matching_paths(raw_roots_of_interest, COMPILED_RAWS, path_root=LOCAL_MTPT / raw_root)
get_and_shuttle_matching_paths(exp_roots_of_interest, COMPILED_COLLAB, path_root=LOCAL_MTPT / exp_root)