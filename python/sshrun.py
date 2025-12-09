#!/usr/bin/env python3
import argparse
import datetime
import os
import shlex
import subprocess
import sys
import tempfile


# Helper function for logging
def _write_sshrun_log_entry(
    log_file_path,
    status_tag, # e.g., "START", "END", "FAILURE", "ERROR"
    host_arg,
    remote_tmp_dir_arg, # The actual remote temp dir or a placeholder
    remote_dir_removed_status_arg,
    command_exit_status_arg,
    output_summary_arg,
    full_output_filename_arg,
    command_arg,
    timestamp_arg
):
    # Escape and quote fields for CSV format
    # Any field value containing a double quote must have that quote doubled,
    # and the entire field must be enclosed in double quotes.
    # We apply this universally for consistency with CSV standard practices.
    _timestamp_arg = str(timestamp_arg).replace('"', '""')
    _status_tag = str(status_tag).replace('"', '""')
    _host_arg = str(host_arg).replace('"', '""')
    _remote_tmp_dir_arg = str(remote_tmp_dir_arg).replace('"', '""')
    _remote_dir_removed_status_arg = str(remote_dir_removed_status_arg).replace('"', '""')
    _command_exit_status_arg = str(command_exit_status_arg).replace('"', '""') # Ensure int is stringified
    _output_summary_arg = str(output_summary_arg).replace('"', '""')
    _full_output_filename_arg = str(full_output_filename_arg).replace('"', '""')
    _command_arg = str(command_arg).replace('"', '""')

    log_entry = (
        f'"{_timestamp_arg}",'
        f'"{_status_tag}",'
        f'"{_host_arg}",'
        f'"{_remote_tmp_dir_arg}",'
        f'"{_remote_dir_removed_status_arg}",'
        f'"{_command_exit_status_arg}",'
        f'"{_output_summary_arg}",'
        f'"{_full_output_filename_arg}",'
        f'"{_command_arg}"\n'
    )
    try:
        # Check if the file is empty to add a header
        file_exists_and_not_empty = os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0

        with open(log_file_path, "a") as f:
            if not file_exists_and_not_empty:
                f.write(
                    "\"Date and Time\","
                    "\"Status\","
                    "\"Remote Host\","
                    "\"Remote Temp Directory\","
                    "\"Removed Status\","
                    "\"Exit Status\","
                    "\"Output Summary\","
                    "\"Full Output File\","
                    "\"Command Executed\"\n"
                )
            f.write(log_entry)
        print(f"[{host_arg}] Log entry ({status_tag}) written to {log_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"[{host_arg}] Error writing to log file {log_file_path}: {e}", file=sys.stderr)


def run_remote_script(host, command, files_to_transfer, files_to_retrieve=None,
                      remote_dir_to_reuse=None, follow_symlinks=True, remote_tmp_parent_dir="/dev/shm"):
    """
    Transfers files to a remote host, runs a command in a temporary directory
    (or a specified existing directory), retrieves results (stdout or specific files),
    and cleans up. It also provides an option to control symlink following during file transfers.

    Args:
        host (str): The SSH host (e.g., 'user@example.com').
        command (str): The command to execute on the remote host.
        files_to_transfer (list): A list of local file paths to transfer.
        files_to_retrieve (list, optional): A list of remote file paths to retrieve
                                           from the temporary directory. Defaults to None.
        remote_dir_to_reuse (str, optional): An existing remote directory to use instead of creating a new one.
                                             If specified, this directory will not be deleted by the script.
        remote_tmp_parent_dir (str, optional): The parent directory on the remote host where the temporary
                                               directory will be created. Defaults to "/dev/shm".
    """
    # --- Variables for tracking state and logging ---
    remote_tmp_dir = "" # Actual remote directory path
    is_new_remote_dir = False
    remote_dirs_to_create = set()
    command_failed = False
    command_exit_status = -1  # Default to -1 if command not run or an error occurs

    # Variables specifically for the log entry, initialized with start-time info or placeholders
    log_file_path = "sshrun.log" # Define log file path once
    current_start_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_remote_tmp_dir = remote_dir_to_reuse if remote_dir_to_reuse else "PENDING" # Set if reusing
    log_remote_dir_removed_status = "PENDING"
    log_command_exit_status = "UNK"
    log_output_summary = "PENDING"
    log_status = "START" # Initial status for the log entry

    # Generate a unique sequential filename for the full command output
    output_file_index = 0
    while True:
        output_filename = f"sshrun_{output_file_index}.out"
        if not os.path.exists(output_filename):
            break
        output_file_index += 1

    try:
        if remote_dir_to_reuse:
            remote_tmp_dir = remote_dir_to_reuse
            log_remote_tmp_dir = remote_tmp_dir # Update log variable
            print(f"[{host}] Reusing remote directory: {remote_tmp_dir}", file=sys.stderr)
        else:
            # 1. Create a temporary directory on the remote host
            print(f"[{host}] Creating temporary directory in {remote_tmp_parent_dir} on remote host...", file=sys.stderr)
            # Using mktemp as `mktemp -d` output path and we need to capture it
            result = subprocess.run(
                ["ssh", host, f"mktemp -d -p {shlex.quote(remote_tmp_parent_dir)}"],
                capture_output=True,
                text=True,
                check=True
            )
            remote_tmp_dir = result.stdout.strip()
            log_remote_tmp_dir = remote_tmp_dir # Update log variable with actual temp dir
            is_new_remote_dir = True
            print(f"[{host}] Remote temporary directory: {remote_tmp_dir}", file=sys.stderr)

        # Now that remote_tmp_dir is known, write the initial "START" log entry
        _write_sshrun_log_entry(
            log_file_path,
            log_status,
            host,
            log_remote_tmp_dir,
            log_remote_dir_removed_status,
            log_command_exit_status,
            log_output_summary,
            output_filename,
            command,
            current_start_date # Use the original start time of the function call
        )

        # 2. Transfer files to the remote temporary directory
        if files_to_transfer:
            print(f"[{host}] Transferring files to {remote_tmp_dir}...", file=sys.stderr)

            file_transfer_details = [] # Stores (local_path, remote_target_path, remote_display_path)

            for local_file in files_to_transfer:
                if not os.path.exists(local_file):
                    print(f"Error: Local file '{local_file}' not found. Skipping.", file=sys.stderr)
                    continue

                # Derive the path to be used on the remote, relative to remote_tmp_dir.
                # This handles both relative and absolute local_file paths by stripping leading slashes
                # to ensure the path starts relative to the remote_tmp_dir.
                remote_relative_path = local_file.lstrip(os.sep)

                # Derive the remote parent directory path
                remote_relative_dir = os.path.dirname(remote_relative_path)
                if remote_relative_dir: # Only add if it's not a file directly in the temp dir root
                    remote_dirs_to_create.add(f"{remote_tmp_dir}/{remote_relative_dir}")

                # For rsync, we construct the destination path explicitly
                remote_target_path = f"{host}:{remote_tmp_dir}/{remote_relative_path}"
                remote_display_path = f"{remote_tmp_dir}/{remote_relative_path}"
                file_transfer_details.append((local_file, remote_target_path, remote_display_path))

            # Create necessary directories on the remote host before rsync
            for remote_dir in remote_dirs_to_create:
                print(f"[{host}] Creating remote directory: {remote_dir}...", file=sys.stderr)
                subprocess.run(
                    ["ssh", host, f"mkdir -p {shlex.quote(remote_dir)}"],
                    check=True,
                    stdout=sys.stderr, # Redirect ssh's stdout to stderr
                    stderr=sys.stderr  # Redirect ssh's stderr to stderr
                )
            rsync_options = ["-a", "--update", "-P", "-h"]
            if follow_symlinks:
                rsync_options.insert(0, "-L") # -L means follow symlinks

            # Now transfer the files using rsync -a (archive mode preserves metadata)
            for local_file, remote_target_path, remote_display_path in file_transfer_details:
                print(f"[{host}] Transferring: {local_file} -> {remote_display_path}", file=sys.stderr)
                subprocess.run(
                    ["rsync"] + rsync_options + [local_file, remote_target_path],
                    check=True,
                    stdout=sys.stderr,  # Redirect rsync's stdout to stderr
                    stderr=sys.stderr   # Redirect rsync's stderr to stderr
                )

        # 3. Execute the command on the remote host within the temporary directory
        print(f"[{host}] Executing command: '{command}' in {remote_tmp_dir}...", file=sys.stderr)
        # Ensure command is properly quoted for shell execution
        remote_command = f"cd {shlex.quote(remote_tmp_dir)} && {command}"
        result = subprocess.run(
            ["ssh", host, remote_command],
            capture_output=True,
            text=True,
            check=False
        )
        command_exit_status = result.returncode
        log_command_exit_status = command_exit_status # Update log variable

        # Write full stdout/stderr to local file
        with open(output_filename, "w") as f_out:
            f_out.write(f"--- Command: {command} ---\n")
            f_out.write(f"--- Remote Host: {host} ---\n")
            f_out.write(f"--- Remote Dir: {remote_tmp_dir} ---\n")
            f_out.write(f"--- Exit Status: {command_exit_status} ---\n\n")
            if result.stdout:
                f_out.write("--- STDOUT ---\n")
                f_out.write(result.stdout)
            if result.stderr:
                f_out.write("\n--- STDERR ---\n")
                f_out.write(result.stderr)
        print(f"[{host}] Full command output saved to {output_filename}", file=sys.stderr)


        if command_exit_status != 0 and result.stderr:
            log_output_summary = result.stderr.split('\n')[0].strip() # Update log variable
        elif result.stdout:
            log_output_summary = result.stdout.split('\n')[0].strip() # Update log variable
        else:
            # If command ran, and neither stdout nor stderr had a summary, then output was effectively empty.
            log_output_summary = "EMPTY"

        if command_exit_status != 0:
            command_failed = True

        # This print statement remains directed to stdout as requested
        print(f"[{host}] Command stdout:", file=sys.stderr)
        print(result.stdout)

        if result.stderr:
            print(f"[{host}] Command stderr:\n{result.stderr}", file=sys.stderr)

        if command_failed:
            print(f"[{host}] Command exited with non-zero status: {result.returncode}", file=sys.stderr)
        # 4. Retrieve specified result files (if any)
        if files_to_retrieve:
            print(f"[{host}] Retrieving files to current working directory...", file=sys.stderr)
            for remote_file in files_to_retrieve:
                remote_source_path = f"{host}:{remote_tmp_dir}/{remote_file}"
                local_destination_path = "." # Retrieve to current working directory, preserving remote structure

                # rsync with -a (archive mode) automatically creates necessary local directories,
                # so explicit os.makedirs for target subdirectories are not required.

                # Display the intended local path for clarity
                intended_local_path = os.path.join(os.getcwd(), remote_file)
                print(f"[{host}] Retrieving: {remote_source_path.split(':')[-1]} -> {intended_local_path}", file=sys.stderr)
                try:
                    rsync_options = ["-a", "--update", "-P", "-h"]
                    if follow_symlinks:
                        rsync_options.insert(0, "-L") # -L means follow symlinks

                    subprocess.run(
                        ["rsync"] + rsync_options + [remote_source_path, local_destination_path],
                        check=True, # Raise CalledProcessError if rsync fails
                        stdout=sys.stderr,  # Redirect rsync's stdout to stderr
                        stderr=sys.stderr   # Redirect rsync's stderr to stderr
                    )
                    print(f"[{host}] Successfully retrieved: {remote_file} to {intended_local_path}", file=sys.stderr)

                except subprocess.CalledProcessError as e:
                    print(f"[{host}] Error retrieving '{remote_file}'. rsync exited with status {e.returncode}.", file=sys.stderr)
                    # The rsync output is already streamed to stderr, so no need to print e.stdout/e.stderr again
                except FileNotFoundError:
                    print(f"[{host}] Error: 'rsync' command not found. Please ensure it's installed and in your PATH.", file=sys.stderr)
                    sys.exit(1)
                except Exception as e:
                    print(f"[{host}] Unexpected error during retrieval of '{remote_file}': {e}", file=sys.stderr)


    except subprocess.CalledProcessError as e:
        log_status = "ERROR" # Set status for logging
        log_command_exit_status = e.returncode # Use the failing command's exit code
        if e.stderr:
            log_output_summary = e.stderr.split('\n')[0].strip()
        print(f"Error during SSH/SCP operation: {e}", file=sys.stderr)
        if e.stdout: # These only capture if subprocess.run was called with capture_output=True
            print(f"Stdout: {e.stdout}", file=sys.stderr)
        if e.stderr: # These only capture if subprocess.run was called with capture_output=True
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        sys.exit(e.returncode) # Exit with the same return code as the failing command
    except FileNotFoundError:
        log_status = "ERROR" # Set status for logging
        log_output_summary = "Command not found (ssh or rsync)"
        print("Error: 'ssh' or 'rsync' command not found. Please ensure OpenSSH client and rsync are installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        log_status = "ERROR" # Set status for logging
        log_output_summary = f"Unexpected error: {str(e).split('\n')[0].strip()}"
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) # Exit with a generic error code for unexpected errors
    finally:
        # 5. Handle remote directory cleanup and write final log entry
        if remote_tmp_dir:
            # Ensure the final remote_tmp_dir is captured, even if an error occurred early
            log_remote_tmp_dir = remote_tmp_dir
            if remote_dir_to_reuse: # Directory was reused, not created by this script
                log_remote_dir_removed_status = "KEPT (reused)"
                print(f"[{host}] Remote directory {remote_tmp_dir} was reused and kept.", file=sys.stderr)
            elif is_new_remote_dir: # It's a new directory created by this script
                should_delete = False
                sys.stdout.flush() # Ensure all previous stdout messages are displayed
                sys.stderr.flush() # Ensure all previous stderr messages are displayed

                try:
                    response = ""
                    try:
                        print(f"[{host}] Remove remote temporary directory {remote_tmp_dir}? [Y/n] ", file=sys.stderr, end='')
                        sys.stderr.flush() # Ensure the prompt is immediately visible
                        response = sys.stdin.readline().strip().lower()
                    except EOFError: # Handles cases where stdin is closed (e.g., piped input), defaults to 'yes'
                        response = ""

                    if response in ('n', 'no'):
                        print(f"[{host}] Remote directory {remote_tmp_dir} kept. To reuse it later, use: "
                              f"--remote-dir {shlex.quote(remote_tmp_dir)}", file=sys.stderr)
                        should_delete = False
                        log_remote_dir_removed_status = "KEPT (user choice)"
                    else: # Default to yes (empty response, EOFError, or any other input)
                        should_delete = True
                        # log_remote_dir_removed_status will be set to "REMOVED" or "REMOVED (interrupted)" below
                except KeyboardInterrupt:
                    print(f"\n[{host}] Interrupted. Defaulting to deleting remote directory {remote_tmp_dir}.", file=sys.stderr)
                    should_delete = True
                    log_remote_dir_removed_status = "REMOVED (interrupted)"

                if should_delete:
                    print(f"[{host}] Cleaning up remote directory {remote_tmp_dir}...", file=sys.stderr)
                    try:
                        subprocess.run(
                            ["ssh", host, f"rm -rf {shlex.quote(remote_tmp_dir)}"],
                            check=True,
                            capture_output=True,
                            text=True
                        )
                        print(f"[{host}] Remote directory removed.", file=sys.stderr)
                        if log_remote_dir_removed_status == "PENDING": # Only set if not already set by interrupt
                           log_remote_dir_removed_status = "REMOVED"
                    except subprocess.CalledProcessError as e:
                        print(f"[{host}] Error removing remote directory: {e.stderr.strip()}", file=sys.stderr)
                        log_remote_dir_removed_status = "KEPT (cleanup failed)" # If deletion failed
                    except Exception as e:
                        print(f"[{host}] Unexpected error during remote cleanup: {e}", file=sys.stderr)
                        log_remote_dir_removed_status = "KEPT (cleanup failed)" # If deletion failed
            # If log_remote_dir_removed_status is still "PENDING" here, it means no cleanup action was taken (e.g. no remote_tmp_dir created)
            elif not remote_tmp_dir: # No remote_tmp_dir was successfully established
                 log_remote_dir_removed_status = "N/A (no remote dir)"

        # Determine final log status if not already set by an exception
        if log_status == "START": # If no exception occurred, means command ran
            if command_failed:
                log_status = "FAILURE"
            else:
                log_status = "END"

        # Write the final log entry with all collected information
        _write_sshrun_log_entry(
            log_file_path,
            log_status,
            host,
            log_remote_tmp_dir,
            log_remote_dir_removed_status,
            log_command_exit_status,
            log_output_summary,
            output_filename,
            command,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Use end time for final entry
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a command on a remote host, transfer files, and retrieve results."
    )
    parser.add_argument("host", help="The SSH host (e.g., user@example.com).")
    parser.add_argument("command", help="The command to execute on the remote host.")
    parser.add_argument(
        "-t", "--transfer", nargs="*", default=[],
        help="Local files to transfer to the remote host's temporary directory."
    )
    parser.add_argument(
        "-r", "--retrieve", nargs="*", default=None,
        help="Remote files to retrieve from the remote temporary directory after command execution. "
             "These will be downloaded to the current working directory."
    )
    parser.add_argument(
        "--remote-dir", type=str, default=None,
        help="Use an existing remote directory instead of creating a new temporary one. "
             "If specified, this directory will not be deleted by the script."
    )
    parser.add_argument(
        "--no-follow-symlinks", action="store_false", dest="follow_symlinks", default=True,
        help="Do not follow symlinks when transferring or retrieving files (rsync's -a without -L). "
             "By default, symlinks are followed (-L)."
    )
    parser.add_argument(
        "--remote-tmp-parent-dir", type=str, default="/dev/shm",
        help="The parent directory on the remote host where the temporary directory will be created. "
             "Defaults to /dev/shm."
    )
    args = parser.parse_args()

    run_remote_script(args.host, args.command, args.transfer, args.retrieve,
                      remote_dir_to_reuse=args.remote_dir,
                      follow_symlinks=args.follow_symlinks,
                      remote_tmp_parent_dir=args.remote_tmp_parent_dir)
