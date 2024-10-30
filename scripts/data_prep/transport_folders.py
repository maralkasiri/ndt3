"""
    Need more space on the server, transporting to cold storage. Run this from storage, not server (no idea how to find storage IP on server)
    Replaced by simple bash
    ssh crc "find /ihome/rgaunt/joy47/projects/context_general_bci/data/runs/ndt3/ -type f -mtime +7 -o -type d | sed 's|/ihome/rgaunt/joy47/projects/context_general_bci/data/runs/ndt3/||'" | rsync -avz --remove-source-files --files-from=- crc:/ihome/rgaunt/joy47/projects/context_general_bci/data/runs/ndt3/ /path/to/local --prune-empty-dirs
"""

import os
import argparse
import subprocess
from datetime import datetime, timedelta

def is_old_folder(folder_path, days_threshold):
    """Check if the folder is older than the specified number of days."""
    current_time = datetime.now()
    folder_time = datetime.fromtimestamp(os.path.getmtime(folder_path))
    return (current_time - folder_time) > timedelta(days=days_threshold)

def process_old_folders(remote_host, remote_root_dir, local_dest_dir, days_threshold, delete):
    """Process old folders on the remote host."""
    # List folders on remote host
    list_command = f"ssh {remote_host} 'ls -d {remote_root_dir}/*'"
    try:
        result = subprocess.run(list_command, shell=True, check=True, capture_output=True, text=True)
        folders = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error listing folders: {e}")
        return

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        breakpoint()
        print(f"Checking folder: {folder_name}")

        # Check folder age
        age_check_command = f"ssh {remote_host} 'python3 -c \"import os, datetime; print((datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(\'{folder_path}\'))).days)\"'"
        try:
            result = subprocess.run(age_check_command, shell=True, check=True, capture_output=True, text=True)
            folder_age = int(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error checking folder age: {e}")
            continue

        if folder_age > days_threshold:
            print(f"Folder {folder_name} is older than {days_threshold} days. Processing...")
            local_folder_path = os.path.join(local_dest_dir, folder_name)
            scp_command = f"scp -r {remote_host}:{folder_path} {local_folder_path}"
            delete_command = f"ssh {remote_host} 'rm -rf {folder_path}'"
            try:
                subprocess.run(scp_command, shell=True, check=True)
                if delete:
                    subprocess.run(delete_command, shell=True, check=True)
                print(f"Successfully transported {folder_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error transporting {folder_name}: {e}")
        else:
            print(f"Folder {folder_name} is not old enough. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process old folders on a remote host")
    parser.add_argument("--remote_host", type=str, help="Remote host (e.g., user@server.com)", default="crc")
    parser.add_argument("--remote_root_dir", type=str, help="Remote root directory containing the folders", default="/ihome/rgaunt/joy47/projects/context_general_bci/data/runs/ndt3")
    parser.add_argument("--local_dest_dir", type=str, help="Local destination directory for transferred folders", default="D:/Data/ndt3_runs/")
    parser.add_argument("--days_threshold", type=int, default=30, help="Age threshold in days")
    parser.add_argument("--delete", action="store_true", help="Delete old folders instead of transferring")

    args = parser.parse_args()

    if not args.delete and not args.local_dest_dir:
        parser.error("--local_dest_dir is required when not using --delete")

    process_old_folders(
        args.remote_host, args.remote_root_dir, args.local_dest_dir,
        args.days_threshold, args.delete
    )