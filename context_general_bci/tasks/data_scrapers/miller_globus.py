import os
import glob
import fnmatch
from pathlib import Path
import argparse

import globus_sdk
from globus_sdk.scopes import TransferScopes

# Shared at limblab/data root, no src path needed
TARGET_ROOT = '/global/homes/j/joelye9/data/ndt3/limblab/' # Nersc DTN

# These are aliases.
# Structure goes: SubjectAlias / YYYYMMDD /  *.[plx, mat,nsx, nev]
SRC_FOLDERS = {
    'Chewie': ['/Chewie_8I2/'],
    'Fish': ['Fish_12H2'],
    'Greyson': ['Greyson_17L2/CerebusData'],
    'Jaco': ['Jaco_8I1'],
    'Jango': ['Jango_12a1'],
    'Keedoo': ['Keedoo_9I3'],
    'Kevin': ['Kevin_12A2'],
    'Mihili': ['Mihili_12A3'],
    'MrT': ['MrT_9I4'],
    'Pedro': ['Pedro_4C2'],
    'Pop': ['Pop_18e3/CerebusData'],
    'Spike': ['Spike_10I3'],
    'Thelonius': ['Thelonius_7H2'],
    'Thor': ['Thor_5E2']
}

# Potential data conflicts
# In "miller" public data, we already have some data from
# Spike 2012
# Pop 2021
# Mihi 2013-2014
# Greyson 2019
# Chewie 2016
# Jango 2015
# Perich Chewie, Jango, Mihili
# Manual ls of files in pretrianing data
dedup_datetime_blacklist = {
    'Chewie': [
        '20160610', '20160710', '20160927', '20160928', '20160929', '20160930',
        '20161006', '20161007', '20161011', '20161012', '20161013', '20161102',
        '20161103', '20161104', '20131003', '20131022', '20131023', '20131031',
        '20131101', '20131203', '20131204', '20131219', '20131220', '20150309',
        '20150311', '20150312', '20150313', '20150319', '20150629', '20150630',
        '20150701', '20150703', '20150706', '20150707', '20150708', '20150709',
        '20150710', '20150713', '20150714', '20150715', '20150716', '20151103',
        '20151104', '20151106', '20151109', '20151110', '20151112', '20151113',
        '20151116', '20151117', '20151119', '20151120', '20151201', '20160909',
        '20160912', '20160914', '20160915', '20160919', '20160921', '20160923',
        '20131009', '20131010', '20131011', '20131028', '20131029', '20131209',
        '20131210', '20131212', '20131213', '20131217', '20131218', '20150316',
        '20150317', '20150318', '20150320'
    ],
    'Mihili': [
        '20140303', '20140403', '20140603', '20140703', '20141702', '20141802',
        '20140203', '20140217', '20140225', '20140228', '20140304', '20140307',
        '20140211', '20140218', '20140227', '20140306', '20131207', '20131208',
        '20140114', '20140115', '20140116', '20140128', '20140129', '20140212',
        '20140214', '20140222', '20140224', '20140626', '20140627', '20140929',
        '20141203', '20150511', '20150512', '20150610', '20150611', '20150612',
        '20150615', '20150616', '20150617', '20150623', '20150625', '20150626',
        '20140221'
    ],
    'Greyson': [
        '20190812', '20190815', '20190819', '20190911', '20190913', '20190918',
        '20190920', '20190923', '20191007'
    ],
    'Jango': [
        '20150730', '20150731', '20150801', '20150805', '20150806', '20150807',
        '20150808', '20150809', '20150820', '20150824', '20150825', '20150826',
        '20150827', '20150828', '20150831', '20150905', '20150906', '20150908',
        '20151029', '20151102', '20160405', '20160406', '20160407'
    ],
    'Spike': [
        '20120821', '20120822', '20120823', '20120824', '20120829', '20120831',
        '20120905', '20120928', '20121015', '20121017', '20121018', '20121023',
        '20121025', '20121105', '20121107', '20121108', '20121109', '20121112'
    ],
    'Tot': [
        '20130819', '20130821', '20130823', '20130903', '20130905', '20130909',
        '20130820', '20130822', '20130830', '20130904', '20130906', '20130910'
    ]
}

# non-conflicts. Perich T -> Tot


BLACKLIST = [
    'stim',
    's1',
    # Try to flag out bci data..
    'decoder',
    'wf',
    'rnn',
    'kf',
    'bad',
]

# Update these with your Globus endpoint IDs
LIMBLAB_ID = 'c808e8f2-0d1f-4309-98f1-6a53152e9de3'  # LimbLab
DESTINATION_ID = '9d6d994a-6d04-11e5-ba46-22000b92c6ec'  # NERSC DTN

# ndt-scrape-script thick client
CLIENT_ID = "9a1dbacd-6360-4513-b0c8-bdb759b02102"
auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

def login_and_get_transfer_client(*, scopes=TransferScopes.all):
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    auth_client.oauth2_start_flow(requested_scopes=scopes)
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")

    auth_code = input("Please enter the code here: ").strip()
    tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    return globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )

def is_blacklisted(subject, date_str):
    if subject in dedup_datetime_blacklist:
        return date_str in dedup_datetime_blacklist[subject]
    return False


def selective_transfer(
    transfer_client,
    source_dir,
    target_dir,
    subject,
    years, # years needed bc Jaco times out in 10 min globus transfer
    neural_patterns='*.nev',
    cov_patterns='*.ns3',
    blacklist=BLACKLIST
):
    transfer_data = globus_sdk.TransferData(transfer_client, LIMBLAB_ID, DESTINATION_ID)

    response = transfer_client.operation_ls(LIMBLAB_ID, path=source_dir)
    for entry in response:
        if entry['type'] == 'dir' and not any(bl in entry['name'].lower() for bl in blacklist):
            src_session_dir = entry['name']  # is basically just the date
            if is_blacklisted(subject, src_session_dir):
                continue
            if not any(src_session_dir.startswith(year) for year in years):
                continue
            src_session_path = os.path.join(source_dir, src_session_dir)
            target_session_path = os.path.join(target_dir, src_session_dir)

            response_nev = transfer_client.operation_ls(LIMBLAB_ID, path=src_session_path)
            for entry_nev in response_nev:
                if (entry_nev['type'] == 'file' and
                    fnmatch.fnmatch(entry_nev['name'], neural_patterns) and
                    not any(bl in entry_nev['name'].lower() for bl in blacklist)):
                    src_neural_file = os.path.join(src_session_path, entry_nev['name'])
                    stem = os.path.splitext(entry_nev['name'])[0]

                    response_cov = transfer_client.operation_ls(LIMBLAB_ID, path=src_session_path)
                    for entry_cov in response_cov:
                        if entry_cov['type'] == 'file' and fnmatch.fnmatch(entry_cov['name'], stem + cov_patterns):
                            target_neural_file = os.path.join(target_session_path, entry_nev['name'])
                            target_cov_file = os.path.join(target_session_path, entry_cov['name'])
                            src_cov_file = os.path.join(src_session_path, entry_cov['name'])

                            print(f"Adding transfer: {src_neural_file} -> {target_neural_file}")
                            transfer_data.add_item(src_neural_file, target_neural_file)
                            transfer_data.add_item(src_cov_file, target_cov_file)
                            break

    task = transfer_client.submit_transfer(transfer_data)
    print(f"Transfer task submitted. Task ID: {task['task_id']}")

def main():
    parser = argparse.ArgumentParser(description="Copy Limblab data using Globus.")
    parser.add_argument('-s', '--subjects', nargs='+', required=True, help="Subjects to scrape. Should be registered in this codefile")
    parser.add_argument('-y', '--years', nargs='+', required=True, help="Years to filter the data (e.g., 2015 2016 2017)")
    args = parser.parse_args()

    transfer_client = login_and_get_transfer_client()

    for subject in args.subjects:
        if subject not in SRC_FOLDERS:
            print(f"Subject {subject} not in list of subjects to scrape.")
            continue

        for subfolder in SRC_FOLDERS[subject]:
            source_dir = subfolder
            target_dir = os.path.join(TARGET_ROOT, subject)

            try:
                selective_transfer(transfer_client, source_dir, target_dir, subject, args.years)
            except globus_sdk.TransferAPIError as err:
                if not err.info.consent_required:
                    raise
                print(
                    "Encountered a ConsentRequired error.\n"
                    "You must login a second time to grant consents.\n\n"
                )
                transfer_client = login_and_get_transfer_client(
                    scopes=err.info.consent_required.required_scopes
                )
                selective_transfer(transfer_client, source_dir, target_dir, subject, args.years)

if __name__ == "__main__":
    main()