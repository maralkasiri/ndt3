# pull pt from nersc

import argparse
import os
import subprocess
import sys

def transfer_folders(remote_host, remote_root, local_root, folders):
    """
    Transfers multiple folders from a remote host to the local machine using scp.

    Args:
        remote_host (str): The remote host in the format user@host.
        remote_root (str): The root directory on the remote host.
        local_root (str): The local root directory where the folders should be saved.
        folders (list): A list of folder names to transfer.
    """
    local_root = os.path.expanduser(local_root)

    for folder in folders:
        remote_path = f"{remote_root}/{folder}"
        local_path = f"{local_root}/{folder}"

        try:
            # Construct the scp command
            scp_command = f"scp -r {remote_host}:{remote_path} {local_path}"

            # Execute the scp command
            result = subprocess.run(scp_command, shell=True, check=True)

            # Check if the command was successful
            if result.returncode == 0:
                print(f"Transfer of folder '{folder}' completed successfully.")
            else:
                print(f"Transfer of folder '{folder}' failed.")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during transfer of folder '{folder}': {e}")
            sys.exit(1)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Transfer multiple folders from a remote host to a local machine using scp.")

    # Add arguments
    parser.add_argument("--remote_host", help="The remote host in the format user@host", default="perl") # Assuming ssh registered
    parser.add_argument("--remote_root", help="The root directory on the remote host", default="projects/ndt3/data/runs/ndt3")
    parser.add_argument("--local_root", help="The local root directory where the folders should be saved", default="~/projects/ndt3/data/runs/ndt3")
    parser.add_argument("folders", nargs='+', help="A list of folder names to transfer")

    # Parse the arguments
    args = parser.parse_args()

    # Transfer the folders
    transfer_folders(args.remote_host, args.remote_root, args.local_root, args.folders)

if __name__ == "__main__":
    main()
