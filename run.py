import argparse
import sys
import time
from datetime import datetime
import subprocess

from src.crash_report.crash_report import send_mail


report_on_completion = True
report_on_failure = True
restart_on_failure = True

INITIAL_BACKOFF = 60            # 1min initial backoff after crash
MAX_BACKOFF = 3600              # 1h max backoff
RESET_BACKOFF_AFTER = 20*60     # reset backoff after 20min


def exponential_backoff(optimistic=False):
    if optimistic:
        yield 0

    bkoff = INITIAL_BACKOFF
    while bkoff < MAX_BACKOFF:
        yield bkoff
        bkoff **= 2

    while True:
        yield MAX_BACKOFF


def verbose_sleep(sec):
    if sec <= 0:
        return

    for remaining in range(sec, -1, -1):
        sys.stdout.write(f"\r{remaining:>6} seconds remaining")
        sys.stdout.flush()
        time.sleep(1)

    # sys.stdout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Either 'train' or 'test'",
                        choices=['train', 'test'])
    parser.add_argument("output_directory",
                        help="Directory of the resulting output")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument("--report_to", help="Email address to which to send crash and completion reports")
    args = parser.parse_args()

    mode = args.mode
    script = f"src/{mode}_model.py"
    config = args.config or f"{mode}_config.json"
    cargs = ["--config_path", config, "--out_dir", args.output_directory]
    command = ["python", script] + cargs
    cr_recipient = args.report_to

    if (report_on_failure or report_on_completion) and cr_recipient is None:
        print("Reports require an email address, exiting.")
        exit()

    backoffs = exponential_backoff()
    backoff = next(backoffs)

    returncode = None
    while returncode != 0:
        print(f"Executing command {command}\n")
        last_start = time.time()
        process = subprocess.Popen(command, encoding='utf-8',
                                   stderr=subprocess.PIPE, bufsize=0)
        _, suberr = process.communicate()
        returncode = process.wait()
        if returncode != 0:
            print(suberr)
            if report_on_failure:
                nowfmt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                message = f"<p style=\"font-family: 'Lucida Console'; text-decoration: underline\">{command} returned {returncode}</p><p style=\"font-family: 'Lucida Console'; white-space: pre\">{suberr}</p>{'Process was automatically restarted.' if restart_on_failure else 'Process will not be restarted.'}"
                send_mail(cr_recipient, f"Crash Report: {nowfmt}", message)
                print(f"Crash report was sent to {cr_recipient}{', restarting process...' if restart_on_failure else ''}")

        if not restart_on_failure:
            break

        if time.time() - last_start >= RESET_BACKOFF_AFTER:
            backoffs = exponential_backoff()
            backoff = next(backoffs)
            print(f"<backoff> was reset after more than {RESET_BACKOFF_AFTER}s of operation")

        print(f"Program crashed! Backing off for {backoff}s...")
        verbose_sleep(backoff)
        backoff = next(backoffs)
        sys.stdout.write(f" -- backoff now set to {backoff}s\n\n")

    if report_on_completion:
        nowfmt = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        send_mail(cr_recipient, f"Completion Report {nowfmt}",
                  f"<p style=\"font-family: 'Lucida Console'; text-decoration: underline\">{command}</p>finished successfully, automatic reporting and restarting is ending.")
