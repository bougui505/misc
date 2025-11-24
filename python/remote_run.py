#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import tempfile


def run_remote_script(host, command, files_to_transfer, files_to_retrieve=None):
    """
    Transfers files to a remote host, runs a command in a temporary directory,
    retrieves results (stdout or specific files), and cleans up.

    Args:
        host (str): The SSH host (e.g., 'user@example.com').
        command (str): The command to execute on the remote host.
        files_to_transfer (list): A list of local file paths to transfer.
        files_to_retrieve (list, optional): A list of remote file paths to retrieve
                                           from the temporary directory. Defaults to None.
    """
    remote_tmp_dir = ""
    local_tmp_dir = ""

    try:
        # 1. Create a temporary directory on the remote host
        print(f"[{host}] Creating temporary directory on remote host...", file=sys.stderr)
        # Using mktemp as `mktemp -d` output path and we need to capture it
        result = subprocess.run(
            ["ssh", host, "mktemp -d -p /dev/shm"],
            capture_output=True,
            text=True,
            check=True
        )
        remote_tmp_dir = result.stdout.strip()
        print(f"[{host}] Remote temporary directory: {remote_tmp_dir}", file=sys.stderr)

        # 2. Transfer files to the remote temporary directory
        if files_to_transfer:
            print(f"[{host}] Transferring files to {remote_tmp_dir}...", file=sys.stderr)
            for local_file in files_to_transfer:
                if not os.path.exists(local_file):
                    print(f"Error: Local file '{local_file}' not found. Skipping.", file=sys.stderr)
                    continue
                remote_path = f"{host}:{remote_tmp_dir}/{os.path.basename(local_file)}"
                subprocess.run(
                    ["scp", local_file, remote_path],
                    check=True,
                    stdout=sys.stderr,  # Redirect scp's stdout to stderr
                    stderr=sys.stderr   # Redirect scp's stderr to stderr
                )
                print(f"[{host}] Transferred: {local_file} -> {remote_path.split(':')[-1]}", file=sys.stderr)

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
        
        # This print statement remains directed to stdout as requested
        print(f"[{host}] Command stdout:\n{result.stdout}")
        
        if result.stderr:
            print(f"[{host}] Command stderr:\n{result.stderr}", file=sys.stderr)
        
        if result.returncode != 0:
            print(f"[{host}] Command exited with non-zero status: {result.returncode}", file=sys.stderr)
            # Optionally, decide to exit or continue based on error handling policy
        
        # 4. Retrieve specified result files (if any)
        if files_to_retrieve:
            local_tmp_dir = tempfile.mkdtemp(prefix="remote_run_local_")
            print(f"[{host}] Retrieving files to local temporary directory: {local_tmp_dir}...", file=sys.stderr)
            for remote_file in files_to_retrieve:
                remote_full_path = f"{host}:{remote_tmp_dir}/{remote_file}"
                local_retrieve_path = os.path.join(local_tmp_dir, os.path.basename(remote_file))
                print(f"[{host}] Retrieving: {remote_full_path.split(':')[-1]} -> {local_retrieve_path}", file=sys.stderr)
                try:
                    subprocess.run(
                        ["scp", remote_full_path, local_retrieve_path],
                        check=True,
                        stdout=sys.stderr,  # Redirect scp's stdout to stderr
                        stderr=sys.stderr   # Redirect scp's stderr to stderr
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[{host}] Warning: Failed to retrieve '{remote_file}': {e}", file=sys.stderr)
            print(f"[{host}] Retrieved files are in: {local_tmp_dir}", file=sys.stderr)
            print(f"[{host}] Please manually remove '{local_tmp_dir}' when done.", file=sys.stderr)


    except subprocess.CalledProcessError as e:
        print(f"Error during SSH/SCP operation: {e}", file=sys.stderr)
        if e.stdout:
            print(f"Stdout: {e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"Stderr: {e.stderr}", file=sys.stderr)
    except FileNotFoundError:
        print("Error: 'ssh' or 'scp' command not found. Please ensure OpenSSH client is installed and in your PATH.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        # 5. Remove the temporary directory on the remote host
        if remote_tmp_dir:
            print(f"[{host}] Cleaning up remote temporary directory {remote_tmp_dir}...", file=sys.stderr)
            try:
                subprocess.run(
                    ["ssh", host, f"rm -rf {shlex.quote(remote_tmp_dir)}"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"[{host}] Remote temporary directory removed.", file=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(f"[{host}] Error removing remote temporary directory: {e.stderr.strip()}", file=sys.stderr)
            except Exception as e:
                print(f"[{host}] Unexpected error during remote cleanup: {e}", file=sys.stderr)
        # Note: local_tmp_dir is not automatically removed, as the user might want to inspect its contents.

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
             "These will be downloaded to a local temporary directory."
    )

    args = parser.parse_args()

    run_remote_script(args.host, args.command, args.transfer, args.retrieve)
