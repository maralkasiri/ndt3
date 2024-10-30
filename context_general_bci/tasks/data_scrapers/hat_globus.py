# https://globus-sdk-python.readthedocs.io/en/stable/examples/minimal_transfer_script/index.html
# Following ex 2, since nersc doesn't allow arbitrary transferring, needs requested scopes.
import os
import glob
import fnmatch
from pathlib import Path

import globus_sdk
from globus_sdk.scopes import TransferScopes

SRC_ROOT = '/all_raw_datafiles_7/'
TARGET_ROOT = '/Users/joelye/data/hatlab/'
TARGET_ROOT = '/global/homes/j/joelye9/data/ndt3/hatlab/' # Nersc DTN
# TARGET_ROOT = '/Users/joelye/data/hatlab/'
# Structure is generally: Monkey / YYYY / YYMMDD (session folders) / *.nsx
SRC_FOLDERS = {
    'Theseus': ['2021'],
    # 'Theseus': ['2021', '2022'],
    # 'Breaux': ['2017', '2018', '2019', '2020', '2021'],
    # 'Hermes': ['direct'], # no years, just individual files
    # 'Jim': ['direct'], 
    # 'Lester': ['2015', '2016', '2017', 'direct'], # 2017/direct have EMG only recordings, not helpful..
}
BLACKLIST = [
    'stim',
    's1',
]

# ndt-scrape-script thick client
CLIENT_ID = "9a1dbacd-6360-4513-b0c8-bdb759b02102"

MACBOOK_ID = '39dda336-e917-11ee-8711-8d3c7f1c8292'
NERSC_DTN_ID = '9d6d994a-6d04-11e5-ba46-22000b92c6ec'
HATLAB_ID = '45e0a630-7184-4565-9ec4-a63b4121ba18'

auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
def login_and_get_transfer_client(*, scopes=TransferScopes.all):
    auth_client.oauth2_start_flow(requested_scopes=scopes)
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")

    auth_code = input("Please enter the code here: ").strip()
    tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

    # return the TransferClient object, as the result of doing a login
    return globus_sdk.TransferClient(
        authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
    )

transfer_client = login_and_get_transfer_client()

# auth_client.oauth2_start_flow(requested_scopes=TransferScopes.all)
# authorize_url = auth_client.oauth2_get_authorize_url()
# print(f"Please go to this URL and login:\n\n{authorize_url}\n")

# auth_code = input("Please enter the code here: ").strip()
# tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
# transfer_tokens = tokens.by_resource_server["transfer.api.globus.org"]

# # construct an AccessTokenAuthorizer and use it to construct the
# # TransferClient
# transfer_client = globus_sdk.TransferClient(
#     authorizer=globus_sdk.AccessTokenAuthorizer(transfer_tokens["access_token"])
# )

# transfer_client = globus_sdk.TransferClient(authorizer=globus_sdk.ClientCredentialsAuthorizer(client, TRANSFER_SCOPE))

# Define your endpoint IDs and directories
source_endpoint_id = HATLAB_ID
target_endpoint_id = NERSC_DTN_ID
def selective_transfer(
    transfer_client,
    source_dir,
    target_dir, 
    neural_patterns='*.nev', 
    cov_patterns='*.ns3',
    blacklist=BLACKLIST,
    use_simple_session_filter=True):
    r"""
        Expected to be called at the year level, where there are a variable number of individual session folders.
        Query directly for datafiles in these folders.
        use_simple_session_filter: Sanitize for ONLY sessions formatted as YYMMDD or YYYYMMDD
        - There are likely miscellaneous datafiles at top level or sublevels, but unclear how we would ingest those systematically. Go for this primary format.
        # Pull the NEV / NS3s only - everything is too large footprint
        
        neural_patterns: Primary neural datafile query
        other_patterns: Other datafiles to transfer (here we target NS3, which contains LFP / Kin, we mainly want Kin)
        # Note the NSX files may have multiple per session. Sometimes synced (different areas), sometimes possibly different blocks (unspecified).
        # We will treat them as separate datapoints in model.
    """
    # Create a transfer task
    transfer_data = globus_sdk.TransferData(transfer_client, source_endpoint_id, target_endpoint_id)
    
    # TODO globus-ify
    # Plaintext logic
    # for src_session_dir in os.listdir(source_dir):
    #     if any(blacklist in src_session_dir.lower() for blacklist in blacklist):
    #         continue
    #     src_session_path = os.path.join(source_dir, src_session_dir)
    #     target_session_path = os.path.join(target_dir, src_session_dir)
    #     for src_neural_file in glob.glob(os.path.join(src_session_path, neural_patterns)):
    #         if any(blacklist in src_neural_file.lower() for blacklist in blacklist):
    #             continue
    #         stem = os.path.splitext(src_neural_file)[0]
    #         src_cov_file = stem + cov_patterns
    #         if os.path.exists(src_cov_file):
    #             # print(f"Transferring {src_neural_file} and {cov_file}")
    #             # transfer
    #             target_neural_file = os.path.join(target_session_path, os.path.basename(src_neural_file))
    #             target_cov_file = os.path.join(target_session_path, os.path.basename(src_cov_file))
    #             transfer_data.add_item(src_neural_file, target_neural_file)
    #             transfer_data.add_item(src_cov_file, target_cov_file)
    
    # Globus logic
    response = transfer_client.operation_ls(source_endpoint_id, path=source_dir)
    for entry in response:
        if entry['type'] == 'dir' and not any(bl in entry['name'].lower() for bl in blacklist):
            if use_simple_session_filter and (len(entry['name']) not in  [6, 8] or not entry['name'].isdigit()):
                continue
            src_session_dir = entry['name']
            src_session_path = os.path.join(source_dir, src_session_dir)
            target_session_path = os.path.join(target_dir, src_session_dir)
            # if not os./path.exists(target_session_path): # Not globus kosher
                # os.makedirs(target_session_path)

            # List neural files in the session directory
            response_nev = transfer_client.operation_ls(source_endpoint_id, path=src_session_path)
            for entry_nev in response_nev:
                if entry_nev['type'] == 'file' and fnmatch.fnmatch(entry_nev['name'], neural_patterns):
                    src_neural_file = os.path.join(src_session_path, entry_nev['name'])
                    stem = os.path.splitext(entry_nev['name'])[0]

                    # Check if the corresponding cov file exists
                    response_cov = transfer_client.operation_ls(source_endpoint_id, path=src_session_path)
                    for entry_cov in response_cov:
                        if entry_cov['type'] == 'file' and fnmatch.fnmatch(entry_cov['name'], stem + cov_patterns):
                            target_neural_file = os.path.join(target_session_path, entry_nev['name'])
                            target_cov_file = os.path.join(target_session_path, entry_cov['name'])
                            src_cov_file = os.path.join(src_session_path, entry_cov['name'])

                            # Add files to the transfer
                            print(f"Adding transfer: {src_neural_file} -> {target_neural_file}")
                            transfer_data.add_item(src_neural_file, target_neural_file)
                            transfer_data.add_item(src_cov_file, target_cov_file)
                            break
    task = transfer_client.submit_transfer(transfer_data)
    print(f"Transfer task submitted. Task ID: {task['task_id']}")    


    
for monkey, years in SRC_FOLDERS.items():
    for year in years:
        if year == 'direct':
            source_dir = os.path.join(SRC_ROOT, monkey)
            target_dir = os.path.join(TARGET_ROOT, monkey)
        else:
            source_dir = os.path.join(SRC_ROOT, monkey, year)
            target_dir = os.path.join(TARGET_ROOT, monkey, year)
        try:
            selective_transfer(transfer_client, source_dir, target_dir)    
        except globus_sdk.TransferAPIError as err:
            # if the error is something other than consent_required, reraise it,
            # exiting the script with an error message
            if not err.info.consent_required:
                raise
            print(
                "Encountered a ConsentRequired error.\n"
                "You must login a second time to grant consents.\n\n"
            )
            transfer_client = login_and_get_transfer_client(
                scopes=err.info.consent_required.required_scopes
            )
            selective_transfer(transfer_client, source_dir, target_dir)    


# Single scratch
# source_dir = '/all_raw_datafiles_7/Theseus/2022/220106/'
# target_dir = '/Users/joelye/data/hatlab/'

# # Patterns for the files
# nev_pattern = '*.nev'
# ns3_pattern = '*.ns3'

# # Collect .nev files
# nev_files = []
# response = transfer_client.operation_ls(source_endpoint_id, path=source_dir)
# for entry in response:
#     if entry['type'] == 'file' and fnmatch.fnmatch(entry['name'], nev_pattern):
#         nev_files.append(entry['name'])

# transfer_data = globus_sdk.TransferData(transfer_client, source_endpoint_id, target_endpoint_id)

# # Check for corresponding .ns3 files
# response = transfer_client.operation_ls(source_endpoint_id, path=source_dir)
# ns3_files = {entry['name']: entry for entry in response if entry['type'] == 'file' and fnmatch.fnmatch(entry['name'], ns3_pattern)}

# for nev_file in nev_files:
#     # Extract the stem (filename without extension)
#     stem = os.path.splitext(nev_file)[0]
#     corresponding_ns3 = stem + '.ns3'
    
#     # If both .nev and .ns3 files exist, add them to the transfer
#     if corresponding_ns3 in ns3_files:
#         source_nev_path = os.path.join(source_dir, nev_file)
#         target_nev_path = os.path.join(target_dir, nev_file)
#         transfer_data.add_item(source_nev_path, target_nev_path)
        
#         source_ns3_path = os.path.join(source_dir, corresponding_ns3)
#         target_ns3_path = os.path.join(target_dir, corresponding_ns3)
#         transfer_data.add_item(source_ns3_path, target_ns3_path)

# # Submit the transfer task
# task = transfer_client.submit_transfer(transfer_data)
# print(f"Transfer task submitted. Task ID: {task['task_id']}")
