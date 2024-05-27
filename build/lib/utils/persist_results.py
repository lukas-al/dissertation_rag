import os
import pickle

"""
Convenience functions to persist the results of the experiements.
Yes I probably don't need to go into this level of detail - but it'll be useful for the future.

Requirements:
- Easily persist all the embedding info and such to a file


"""


def save_results(
    experiment_type: str,
    uuid: str,
    persist_objects: dict,
):
    """Save the results of an experiment to a new folder in the filesystem
    The experiments are grouped by experiment_type, which creates a top level folder.
    Runs are stored as individual sub-folders, with the uuid as the folder name.
    The persist_objects are saved to the sub-folders as individual files (usually as pickles).

    Args:
        experiment_type (str): The type of experiment
        uuid (str): The unique identifier for the run
        persist_objects (dict): The objects to be persisted
    """

    print("Persisting results for experiment:", experiment_type, "with uuid:", uuid)

    # Get the project root directory
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Create the top level folder for the experiment type
    experiment_folder = os.path.join(project_root, "results", experiment_type)
    os.makedirs(experiment_folder, exist_ok=True)

    # Create the sub-folder for the run
    run_folder = os.path.join(experiment_folder, uuid)

    # Check if the folder already exists
    if os.path.exists(run_folder):
        # Find the next available folder name
        i = 1
        while True:
            new_folder_name = f"{uuid}_{i}"
            new_run_folder = os.path.join(experiment_folder, new_folder_name)
            if not os.path.exists(new_run_folder):
                run_folder = new_run_folder
                break
            i += 1

    os.makedirs(run_folder, exist_ok=True)

    # Save each object as a separate file in the run folder
    for name, obj in persist_objects.items():
        file_path = os.path.join(run_folder, f"{name}.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    print("------")
