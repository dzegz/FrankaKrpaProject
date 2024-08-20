import subprocess
import time
import os
import pyautogui
import signal
import psutil

# Paths to your bash scripts
start_script_1 = './start_camera_kinect_azure.sh'
start_script_2 = './run_main.sh'

# Conda environment name
conda_env = 'aldin-cloth'

# Path to the socket file
socket_file = '/tmp/echo.sock'

# Script names for pkill
script_names = [start_script_1, start_script_2]

def cleanup_stale_socket():
    """Check for existing/stale socket files and remove them if necessary."""
    if os.path.exists(socket_file):
        try:
            os.remove(socket_file)
            print(f"Removed stale socket file: {socket_file}")
        except Exception as e:
            print(f"Failed to remove socket file {socket_file}: {e}")
    else:
        print(f"No stale socket file found at {socket_file}")

def run_bash_script_in_terminal_with_conda(script_path):
    """Runs a bash script in a new terminal window within a Conda environment."""
    try:
        # Open gnome-terminal and run commands in sequence
        command = f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env} && cd docker && bash {script_path} && exec bash'
        proc = subprocess.Popen([
            'gnome-terminal', '--', 'bash', '-c', command
        ])
        print(f"Started {script_path} in a new terminal with Conda environment {conda_env}")
    except Exception as e:
        print(f"Failed to start {script_path} in a new terminal: {e}")

def simulate_space_key():
    """Simulates the SPACE key press using pyautogui."""
    time.sleep(5)  # Adjust delay to ensure the script is ready before pressing SPACE
    pyautogui.press('space') 
    print("Simulated SPACE key press")

def terminate_processes(script_name):
    """Terminate all processes related to the script."""
    try:
        # Terminate processes using pkill based on script name
        subprocess.run(['pkill', '-f', script_name], check=True)
        print(f"Terminated processes related to {script_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to terminate processes for {script_name}: {e}")

def main():
    # Step 1: Clean up any stale socket files
    time.sleep(2)
    cleanup_stale_socket()

    # Step 2: Start the first bash script in a new terminal with conda environment
    run_bash_script_in_terminal_with_conda(start_script_1)

    # Step 3: Wait for 7 seconds before starting the second script
    time.sleep(7)

    # Step 4: Start the second bash script in a new terminal with conda environment
    run_bash_script_in_terminal_with_conda(start_script_2)

    # Step 5: Wait a bit to ensure the second script is running and ready
    time.sleep(5)

    # Step 6: Simulate the "SPACE" key press for the second script
    simulate_space_key()

    # Step 7: Wait for a bit to let the commands finish
    time.sleep(5)  # Adjust this delay if necessary

    # Step 8: Close both terminals
    print("Closing the terminals...")
    for script_name in script_names:
        terminate_processes(script_name)

    # Step 9: Cycle completion message
    print("Picture taken and stored.")

if __name__ == "__main__":
    main()
