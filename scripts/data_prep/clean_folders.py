# We're storing too many ckpts. Remove some, that have been removed on remote dashboard, or otherwise automatically detected as bad.

import os
import shutil
import wandb
from datetime import datetime

def folder_exists_on_wandb(run_id, project, entity, status="finished"):
    """Check if a run exists on wandb."""
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        if run.state == status or run.state == "running":
            return True # keep
        return False
    except wandb.errors.CommError:
        return False
    except wandb.errors.Error as e:
        print(f"Error while checking run {run_id}: {e}")
        return False

def is_within_date_range(folder_path, start_date, end_date):
    """Check if the folder was created or modified within the date range."""
    folder_time = datetime.fromtimestamp(os.path.getmtime(folder_path))
    return start_date <= folder_time <= end_date

def remove_folder_if_run_not_exists_or_run_not_in_status(root_dir, project, entity, start_date, end_date,
                                                     status="finished"):
    """Iterate through folders in root_dir and remove those whose corresponding runs do not exist on wandb, within a certain date range.
        Or if the wandb status is bad // (e.g. failed, not finished - not the final run we care about.)
    """
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            if is_within_date_range(folder_path, start_date, end_date):
                print(f"Checking folder: {folder_name}")
                if not folder_exists_on_wandb(folder_name, project, entity):
                    print(f"Run {folder_name} flagged in bad wandb state and is within date range. Removing folder...")
                    shutil.rmtree(folder_path)
                else:
                    pass
                    # print(f"Run {folder_name} exists on wandb. Keeping folder.")
            else:
                pass
                # print(f"Folder {folder_name} is not within the date range. Skipping.")

if __name__ == "__main__":
    root_directory = "./data/runs/ndt3"  # Root directory containing the folders
    root_directory = "/ihome/rgaunt/joy47/scratch/data/runs/ndt3"  # Root directory containing the folders
    wandb_project = "ndt3"
    wandb_entity = "joelye9"
    # Define the start and end date for the filter
    # start_date = datetime(2024, 8, 20)  # Start date
    # end_date = datetime(2024, 7, 20)  # End date
    start_date = datetime(2024, 7, 18)  # Start date
    # end_date = datetime(2024, 8, 21)  # End date
    # start_date = datetime(2024, 8, 21)  # Start date
    end_date = datetime(2024, 12, 1)  # Start date

    remove_folder_if_run_not_exists_or_run_not_in_status(root_directory, wandb_project, wandb_entity, start_date, end_date)
