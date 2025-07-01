import os
import argparse
from pathlib import Path

def get_file_map(directory):
    """Gets a dictionary mapping filename stems to their full paths."""
    file_map = {}
    if not os.path.isdir(directory):
        print(f"Warning: Directory not found: {directory}")
        return file_map
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            stem = Path(f).stem
            file_map[stem] = os.path.join(directory, f)
    return file_map

def check_and_sync_directories(base_dir, other_dirs, sync=False):
    """
    Checks for filename consistency between a base directory and other directories.
    If sync is True, it deletes files from other_dirs that are not in base_dir.
    """
    base_files = get_file_map(base_dir)
    base_stems = set(base_files.keys())
    print(f"Base directory '{base_dir}' has {len(base_stems)} files.")
    if '.DS_Store' in base_stems:
        base_stems.remove('.DS_Store')


    for directory in other_dirs:
        print(f"\nProcessing directory: '{directory}'")
        other_files_map = get_file_map(directory)
        other_stems = set(other_files_map.keys())
        print(f"'{directory}' has {len(other_stems)} files.")

        # --- Check for files in base but not in other ---
        missing_in_other = base_stems - other_stems
        if missing_in_other:
            print(f"  INFO: {len(missing_in_other)} files are in '{base_dir}' but NOT in '{directory}'.")
            # Example files
            for filename in sorted(list(missing_in_other))[:3]:
                print(f"    - {filename}")
            if len(missing_in_other) > 3:
                print(f"    ... and {len(missing_in_other) - 3} more.")


        # --- Check for and handle files in other but not in base ---
        extra_in_other = other_stems - base_stems
        if extra_in_other:
            print(f"  INFO: {len(extra_in_other)} files are in '{directory}' but NOT in '{base_dir}'.")
            if sync:
                print(f"  SYNCING: Deleting {len(extra_in_other)} extra files...")
                deleted_count = 0
                for stem in extra_in_other:
                    file_to_delete = other_files_map.get(stem)
                    if file_to_delete:
                        try:
                            os.remove(file_to_delete)
                            deleted_count += 1
                        except OSError as e:
                            print(f"    ERROR: Could not delete {file_to_delete}: {e}")
                print(f"  SUCCESS: Deleted {deleted_count} files.")
            else:
                print("  (Run with --action sync to delete these files)")
                # Example files
                for filename in sorted(list(extra_in_other))[:3]:
                    print(f"    - {filename}")
                if len(extra_in_other) > 3:
                    print(f"    ... and {len(extra_in_other) - 3} more.")

        if not missing_in_other and not extra_in_other:
            print("  Directories are consistent.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for and sync file consistency across data directories based on filenames (ignoring extensions)."
    )
    parser.add_argument(
        "--action",
        type=str,
        default="check",
        choices=["check", "sync"],
        help="The action to perform. 'check' (default) only reports differences. 'sync' deletes extra files."
    )
    args = parser.parse_args()

    data_folder = "data"
    base_directory = os.path.join(data_folder, "tof")
    directories_to_process = [
        os.path.join(data_folder, "confidence"),
        os.path.join(data_folder, "depth"),
        os.path.join(data_folder, "original_pose"),
        os.path.join(data_folder, "pose"),
        os.path.join(data_folder, "webcam"),
    ]

    print(f"Starting process with action: '{args.action}'")
    check_and_sync_directories(base_directory, directories_to_process, sync=(args.action == "sync"))
    print("\nProcess finished.")