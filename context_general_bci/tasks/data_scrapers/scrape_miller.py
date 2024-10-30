r"""
    Assuming Limblab is mounted and visible at input path.
    Extract the relevant monkey's NS3 and NEVs.
"""
from pathlib import Path
import os
import subprocess
import argparse
from datetime import datetime

# path
# /mnt/limblab/limblab/data/limblab/MonkeyName/Year/ExperimentDate/NS3/NEV
# fsmresfiles:Basic_Sciences/L_MillerLab/data/<Monkey_XXXX>/YYYYMMDD/ nev / ns3 files
# TODO add a size threshold - delist nevs <14KB, no neural data in there, likely.
# Dedup efforts: Jango, Mihili, Chewie, Tot - are in Perich's release

r"""
    TODO:
    - Dedup
    - ..
"""

# Monkeys cleared with Lee to try to scrape, that have at least an M1 implant, avoiding stim keywords.
# Get to root of where NSX files are
target_subject_roots = {
    'Chewie': 'Chewie_8l2',
    # 'Fashizzle': 'Fashizzle_3F1', # seemingly all plexon
    'Fish': 'Fish_12H2', # likely
    'Greyson': 'Greyson_17L2/CerebusData', # likely
    'Jaco': 'Jaco_8l1', # has ns2s...
    'Jango': 'Jango_12a1', # maybe
    'Keedoo': 'Keedoo_9l3', # suspect
    'Kevin': 'Kevin_12A2',
    'Mihili': 'Mihili_12A3',
    'Mini': '', # plx in some part
    'MrT': 'MrT_9l4',
    'Pedro': 'Pedro_4C2',
    'Pop': 'Pop_18e3/CerebusData', # WIP need to finish. Looks like cage data?
    'Spike': 'Spike_10l3',
    'Thelonius': 'Thelonius_7H2', # lots of plx, but not all
    'Thor': 'Thor_5E2' # thor 5e2 # 1Dh is his
}

# Avoid these datetimes for these monkeys, they've been released publicly and are already in pretraining pool.
dedup_datetime_blacklist = {
    # Chewie_CO_20160610_M1.mat  Chewie_CO_20162909_M1.mat  Mihili_CO_20140403_M1.mat  Mihili_CO_20140703_M1.mat  Mihili_CO_20141802_M1.mat
    # Chewie_CO_20160710_M1.mat  Mihili_CO_20140303_M1.mat  Mihili_CO_20140603_M1.mat  Mihili_CO_20141702_M1.mat
    # full-mihi-03032014.mat  full-mihi-03062014.mat

    # Miller dump - several Miller collisions already.
    # Mihili_20140203_001.mat  Mihili_20140217_001.mat  Mihili_20140225_001.mat  Mihili_20140228_001.mat  Mihili_20140304_001.mat  Mihili_20140307_001.mat
    # Mihili_20140211_001.mat  Mihili_20140218_001.mat  Mihili_20140227_001.mat  Mihili_20140303_001.mat  Mihili_20140306_001.mat
    # Chewie_20160927_001.mat  Chewie_20160929_001.mat  Chewie_20161006_001.mat  Chewie_20161011_001.mat  Chewie_20161013_001.mat  Chewie_20161103_001.mat
    # Chewie_20160928_001.mat  Chewie_20160930_001.mat  Chewie_20161007_001.mat  Chewie_20161012_001.mat  Chewie_20161102_001.mat  Chewie_20161104_001.mat
    # Greyson_20190812_Key_001.mat  Greyson_20190819_Key_001.mat  Greyson_20190913_Key_001.mat  Greyson_20190920_Key_001.mat  Greyson_20191007_Key_001.mat
    # Greyson_20190815_Key_001.mat  Greyson_20190911_Key_001.mat  Greyson_20190918_Key_001.mat  Greyson_20190923_Key_001.mat
    # Jango_20150730_001.mat  Jango_20150806_001.mat  Jango_20150820_001.mat  Jango_20150827_001.mat  Jango_20150906_001.mat
    # Jango_20150731_001.mat  Jango_20150807_001.mat  Jango_20150824_001.mat  Jango_20150828_001.mat  Jango_20150908_001.mat
    # Jango_20150801_001.mat  Jango_20150808_001.mat  Jango_20150825_001.mat  Jango_20150831_001.mat  Jango_20151029_001.mat
    # Jango_20150805_001.mat  Jango_20150809_001.mat  Jango_20150826_001.mat  Jango_20150905_001.mat  Jango_20151102_001.mat
    # Mihili_20131207_001_RT.mat  Mihili_20140115_001_RT.mat  Mihili_20140129_001_RT.mat  Mihili_20140222_001_RT.mat
    # Mihili_20131208_001_RT.mat  Mihili_20140116_001_RT.mat  Mihili_20140212_001_RT.mat  Mihili_20140224_001_RT.mat
    # Mihili_20140114_001_RT.mat  Mihili_20140128_001_RT.mat  Mihili_20140214_001_RT.mat
    # Spike_20120821_001.mat  Spike_20120829_001.mat  Spike_20121015_001.mat  Spike_20121025_001.mat  Spike_20121109_001.mat
    # Spike_20120822_001.mat  Spike_20120831_001.mat  Spike_20121017_001.mat  Spike_20121105_001.mat  Spike_20121112_001.mat
    # Spike_20120823_003.mat  Spike_20120905_001.mat  Spike_20121018_001.mat  Spike_20121107_001.mat
    # Spike_20120824_001.mat  Spike_20120928_001.mat  Spike_20121023_002.mat  Spike_20121108_001.mat


    # Perich dump, where M is Mihili, C is Chewie, J is Jango, T is Tot
    # sub-M_ses-CO-20140203_behavior+ecephys.nwb
    # sub-M_ses-CO-20140217_behavior+ecephys.nwb
    # sub-M_ses-CO-20140218_behavior+ecephys.nwb
    # sub-M_ses-CO-20140303_behavior+ecephys.nwb
    # sub-M_ses-CO-20140304_behavior+ecephys.nwb
    # sub-M_ses-CO-20140306_behavior+ecephys.nwb
    # sub-M_ses-CO-20140307_behavior+ecephys.nwb
    # sub-M_ses-CO-20140626_behavior+ecephys.nwb
    # sub-M_ses-CO-20140627_behavior+ecephys.nwb
    # sub-M_ses-CO-20140929_behavior+ecephys.nwb
    # sub-M_ses-CO-20141203_behavior+ecephys.nwb
    # sub-M_ses-CO-20150511_behavior+ecephys.nwb
    # sub-M_ses-CO-20150512_behavior+ecephys.nwb
    # sub-M_ses-CO-20150610_behavior+ecephys.nwb
    # sub-M_ses-CO-20150611_behavior+ecephys.nwb
    # sub-M_ses-CO-20150612_behavior+ecephys.nwb
    # sub-M_ses-CO-20150615_behavior+ecephys.nwb
    # sub-M_ses-CO-20150616_behavior+ecephys.nwb
    # sub-M_ses-CO-20150617_behavior+ecephys.nwb
    # sub-M_ses-CO-20150623_behavior+ecephys.nwb
    # sub-M_ses-CO-20150625_behavior+ecephys.nwb
    # sub-M_ses-CO-20150626_behavior+ecephys.nwb
    # sub-M_ses-RT-20140114_behavior+ecephys.nwb
    # sub-M_ses-RT-20140115_behavior+ecephys.nwb
    # sub-M_ses-RT-20140116_behavior+ecephys.nwb
    # sub-M_ses-RT-20140214_behavior+ecephys.nwb
    # sub-M_ses-RT-20140221_behavior+ecephys.nwb
    # sub-M_ses-RT-20140224_behavior+ecephys.nwb

    # sub-C_ses-CO-20131003_behavior+ecephys.nwb  sub-C_ses-CO-20150713_behavior+ecephys.nwb  sub-C_ses-CO-20160929_behavior+ecephys.nwb
    # sub-C_ses-CO-20131022_behavior+ecephys.nwb  sub-C_ses-CO-20150714_behavior+ecephys.nwb  sub-C_ses-CO-20161005_behavior+ecephys.nwb
    # sub-C_ses-CO-20131023_behavior+ecephys.nwb  sub-C_ses-CO-20150715_behavior+ecephys.nwb  sub-C_ses-CO-20161006_behavior+ecephys.nwb
    # sub-C_ses-CO-20131031_behavior+ecephys.nwb  sub-C_ses-CO-20150716_behavior+ecephys.nwb  sub-C_ses-CO-20161007_behavior+ecephys.nwb
    # sub-C_ses-CO-20131101_behavior+ecephys.nwb  sub-C_ses-CO-20151103_behavior+ecephys.nwb  sub-C_ses-CO-20161011_behavior+ecephys.nwb
    # sub-C_ses-CO-20131203_behavior+ecephys.nwb  sub-C_ses-CO-20151104_behavior+ecephys.nwb  sub-C_ses-CO-20161013_behavior+ecephys.nwb
    # sub-C_ses-CO-20131204_behavior+ecephys.nwb  sub-C_ses-CO-20151106_behavior+ecephys.nwb  sub-C_ses-CO-20161021_behavior+ecephys.nwb
    # sub-C_ses-CO-20131219_behavior+ecephys.nwb  sub-C_ses-CO-20151109_behavior+ecephys.nwb  sub-C_ses-RT-20131009_behavior+ecephys.nwb
    # sub-C_ses-CO-20131220_behavior+ecephys.nwb  sub-C_ses-CO-20151110_behavior+ecephys.nwb  sub-C_ses-RT-20131010_behavior+ecephys.nwb
    # sub-C_ses-CO-20150309_behavior+ecephys.nwb  sub-C_ses-CO-20151112_behavior+ecephys.nwb  sub-C_ses-RT-20131011_behavior+ecephys.nwb
    # sub-C_ses-CO-20150311_behavior+ecephys.nwb  sub-C_ses-CO-20151113_behavior+ecephys.nwb  sub-C_ses-RT-20131028_behavior+ecephys.nwb
    # sub-C_ses-CO-20150312_behavior+ecephys.nwb  sub-C_ses-CO-20151116_behavior+ecephys.nwb  sub-C_ses-RT-20131029_behavior+ecephys.nwb
    # sub-C_ses-CO-20150313_behavior+ecephys.nwb  sub-C_ses-CO-20151117_behavior+ecephys.nwb  sub-C_ses-RT-20131209_behavior+ecephys.nwb
    # sub-C_ses-CO-20150319_behavior+ecephys.nwb  sub-C_ses-CO-20151119_behavior+ecephys.nwb  sub-C_ses-RT-20131210_behavior+ecephys.nwb
    # sub-C_ses-CO-20150629_behavior+ecephys.nwb  sub-C_ses-CO-20151120_behavior+ecephys.nwb  sub-C_ses-RT-20131212_behavior+ecephys.nwb
    # sub-C_ses-CO-20150630_behavior+ecephys.nwb  sub-C_ses-CO-20151201_behavior+ecephys.nwb  sub-C_ses-RT-20131213_behavior+ecephys.nwb
    # sub-C_ses-CO-20150701_behavior+ecephys.nwb  sub-C_ses-CO-20160909_behavior+ecephys.nwb  sub-C_ses-RT-20131217_behavior+ecephys.nwb
    # sub-C_ses-CO-20150703_behavior+ecephys.nwb  sub-C_ses-CO-20160912_behavior+ecephys.nwb  sub-C_ses-RT-20131218_behavior+ecephys.nwb
    # sub-C_ses-CO-20150706_behavior+ecephys.nwb  sub-C_ses-CO-20160914_behavior+ecephys.nwb  sub-C_ses-RT-20150316_behavior+ecephys.nwb
    # sub-C_ses-CO-20150707_behavior+ecephys.nwb  sub-C_ses-CO-20160915_behavior+ecephys.nwb  sub-C_ses-RT-20150317_behavior+ecephys.nwb
    # sub-C_ses-CO-20150708_behavior+ecephys.nwb  sub-C_ses-CO-20160919_behavior+ecephys.nwb  sub-C_ses-RT-20150318_behavior+ecephys.nwb
    # sub-C_ses-CO-20150709_behavior+ecephys.nwb  sub-C_ses-CO-20160921_behavior+ecephys.nwb  sub-C_ses-RT-20150320_behavior+ecephys.nwb
    # sub-C_ses-CO-20150710_behavior+ecephys.nwb  sub-C_ses-CO-20160923_behavior+ecephys.nwb

    # sub-J_ses-CO-20160405_behavior+ecephys.nwb  sub-J_ses-CO-20160406_behavior+ecephys.nwb  sub-J_ses-CO-20160407_behavior+ecephys.nwb
    # sub-T_ses-CO-20130819_behavior+ecephys.nwb  sub-T_ses-CO-20130905_behavior+ecephys.nwb  sub-T_ses-RT-20130830_behavior+ecephys.nwb
    # sub-T_ses-CO-20130821_behavior+ecephys.nwb  sub-T_ses-CO-20130909_behavior+ecephys.nwb  sub-T_ses-RT-20130904_behavior+ecephys.nwb
    # sub-T_ses-CO-20130823_behavior+ecephys.nwb  sub-T_ses-RT-20130820_behavior+ecephys.nwb  sub-T_ses-RT-20130906_behavior+ecephys.nwb
    # sub-T_ses-CO-20130903_behavior+ecephys.nwb  sub-T_ses-RT-20130822_behavior+ecephys.nwb  sub-T_ses-RT-20130910_behavior+ecephys.nwb

    'Chewie': [

    ],
}


def scrape_subject(subject, src, dst):
    r"""
        For tractability, we will only pull the obviously legible data.
        That is, paired NS3/NEV in 1 or 2-deep subdirs, i.e. experiments on a given day.
        - 1 deep - per day
        - 2 deep - grouped into some other directory, e.g. year
    """

    # Check if subject is in list
    if subject not in target_subject_roots:
        print(f"Subject {subject} not in list of subjects to scrape.")
    subject_root = Path(src) / target_subject_roots.get(subject)
    subject_dst = Path(dst) / subject

    # If year, recurse

    # else, we have a raw directory. Look for paired NS3/NEV files. Establish identity (subject / datetime), check blacklist, and move under that identity.


def main():
    parser = argparse.ArgumentParser(description="Copy mounted limblab data to another storage system.")
    parser.add_argument('--src_root', help="Root directory of Limblab data.", type=str)
    parser.add_argument('--dst_root', help="Root directory of destination storage. Should end with data/limblab. Suggested on CRC.", type=str)
    # The rest are subjects
    parser.add_argument('subjects', nargs='+', help="Subjects to scrape. Should be registered in this codefile")
    args = parser.parse_args()

    # mkdir if not made, not dst is on another server, so will need special call

    for subject in args.subjects:
        scrape_subject(subject, args.src_root, args.dst_root)

if __name__ == "__main__":
    main()
