import argparse
from datetime import datetime
import subprocess

from src.crash_report.crash_report import send_mail


report_on_success = True
report_on_failure = True
restart_on_failure = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Either 'train' or 'test'",
                        choices=['train', 'test'])
    parser.add_argument("output_directory",
                        help="Directory of the resulting output")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument(
        "--report_to",
        help="Email address to which to send crash and completion reports")

    args = parser.parse_args()
    mode = args.mode
    script = f"src/{mode}_model.py"
    config = args.config or f"{mode}_config.json"
    cargs = ["--config_path", config, "--out_dir", args.output_directory]
    command = ["python", script] + cargs
    cr_recipient = args.report_to

    if (report_on_failure or report_on_success) and cr_recipient is None:
        print("Reports require an email address, exiting.")
        exit()

    returncode = None
    while returncode != 0:
        print(f"Executing command {command}\n")
        process = subprocess.Popen(command, encoding='utf-8',
                                   stderr=subprocess.PIPE, bufsize=0)
        _, suberr = process.communicate()
        returncode = process.wait()
        if returncode != 0:
            print(suberr)
            if report_on_failure:
                nowfmt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                message = f"<p style=\"font-family: 'Lucida Console'; text-decoration: underline\">{command}</p><p style=\"font-family: 'Lucida Console'; white-space: pre\">{suberr}</p>Process was automatically restarted."
                send_mail(cr_recipient, f"Crash Report: {nowfmt}", message)
                print(f"Crash report was sent to {cr_recipient}{', restarting process...' if restart_on_failure else ''}")

        if not restart_on_failure:
            break

    if report_on_success:
        nowfmt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        send_mail(cr_recipient, f"Completion Report {nowfmt}",
                  f"<p style=\"font-family: 'Lucida Console'; text-decoration: underline\">{command}</p>finished successfully, automatic reporting and restarting is ending.")
