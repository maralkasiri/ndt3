import os
import subprocess

sessions = [
    ("_", "P4", "Lab", 44, [1], 0), # RL
    ("_", "P4", "Lab", 31, [1], 0), # RL
    ("_", "P4", "Lab", 40, [1], 0), # RL
    ("02/03/22", "P3", "Home", 32, [5,7,9,10], 2),
    ("02/03/22", "P3", "Home", 32, [5,7,9,10], 2),
    ("02/03/22", "P3", "Home", 32, [5,7,9,10], 2),
    ("02/10/22", "P3", "Home", 33, [3,6,8,10,11,12], 2),
    ("02/18/22", "P3", "Home", 34, [3,7,9,10], 2),
    ("02/24/22", "P3", "Home", 35, [2,3,5,6,8,9,11,12], 4),
    ("04/28/22", "P3", "Home", 52, [3,4,7,8,9,10,11], 4),
    ("01/19/23", "P3", "Home", 108, [3,4,7,8,11,12,13,14], 4),
    ("01/24/22", "P2", "Lab", 1761, [3,5,7,9,10], 2),
    ("02/07/22", "P2", "Lab", 1767, [2,3,5,6,9,10], 4),
    ("02/28/22", "P2", "Lab", 1769, [3,4,7,8,11,12], 4),
    ("04/13/22", "P2", "Lab", 1776, [3,4,8,9,13,14], 2),
    ("04/27/22", "P2", "Lab", 1778, [6,7,11,12,16,17], 2),
    ("01/11/23", "P2", "Lab", 1900, [3,4,7,8,10,11,12,13], 4),
    ("02/01/23", "P2", "Lab", 1907, [2,5,8,11,13], 2),
    ("02/20/23", "P2", "Lab", 1918, [2,6], 2),
    ("05/15/23", "P2", "Lab", 1965, [3,10], 2),
    ("08/10/23", "P2", "Lab", 2004, [3,7], 2),
    ("08/28/23", "P2", "Lab", 2011, [3,8], 2),
    ("09/06/23", "P2", "Lab", 2016, [2,5], 2),
    ("06/02/23", "P4", "Lab", 10, [4,8], 2),
    ("08/16/23", "P4", "Lab", 36, [3,4], 2),
]

def generate_scp_commands(sessions):
    base_path = "crc:projects/context_general_bci/data/pitt_broad/"
    for date, person, location, session, sets, ramp in sessions:
        session_type = "Lab" if location == "Lab" else "Home"
        session_id = f"{person}{session_type}_session_{session}_set_"
        regex = f"{base_path}{session_id}*"
        scp_command = f"scp {regex} ."
        # print(scp_command)
        subprocess.run(scp_command, shell=True)

generate_scp_commands(sessions)