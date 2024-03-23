import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_subject(subject):
    with open(f'logs/{subject}_stdout.log', 'w') as stdout_file:
        subprocess.run(["python", "MPI_corr_19Sep23_parallel.py", subject], stdout=stdout_file)

if __name__ == "__main__":
    # Create logs folder if not exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Get the list of subjects
    subjects = [subject for subject in os.listdir('/scratch/MPI-LEMON/freesurfer/subjects/inspected') if subject.startswith("sub")]

    # Initialize the ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit the subjects for processing and collect the Future objects
        futures = {executor.submit(run_subject, subject): subject for subject in subjects}

        # Iterate over completed futures and check for exceptions
        for future in as_completed(futures):
            subject = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"An exception occurred in subject {subject}: {e}")
