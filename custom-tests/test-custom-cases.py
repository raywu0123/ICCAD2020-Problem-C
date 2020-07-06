import os
import ntpath
import shutil
import sys
import tempfile
import subprocess
from argparse import ArgumentParser


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run_test(test_dir: str, standard_cell_library_path: str, saving_directory: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_intermediate_file(test_dir, tmpdir, standard_cell_library_path)
        run_simulation(test_dir, tmpdir, saving_directory)
        run_validation(test_dir, tmpdir)


def generate_intermediate_file(test_dir: str, tmpdir: str, standard_cell_library_path: str):
    exec_path = os.path.join(os.getcwd(), "GraphPreprocessing.py")
    gv_path = os.path.join(test_dir, "gv.gv")
    sdf_path = os.path.join(test_dir, "sdf.sdf")
    intermediate_path = os.path.join(tmpdir, "intermediate.txt")
    subprocess.call(
        ["python", exec_path, gv_path, sdf_path, standard_cell_library_path, intermediate_path],
        stdout=open(os.devnull, "wb"),
        stderr=open(os.devnull, "wb"),
    )


def run_simulation(test_dir: str, tmpdir: str, saving_directory: str):
    exec_path = os.path.join(os.getcwd(), "build/GPUSimulator")
    intermediate_path = os.path.join(tmpdir, "intermediate.txt")
    input_vcd_path = os.path.join(test_dir, "input.vcd")
    output_vcd_path = os.path.join(tmpdir, "out.vcd")
    subprocess.call(
        [exec_path, intermediate_path, input_vcd_path, "VCD", output_vcd_path],
        stdout=open(os.devnull, "wb"),
        stderr=open(os.devnull, "wb"),
    )

    if saving_directory is not None:
        case_name = ntpath.basename(test_dir)
        shutil.copy2(output_vcd_path, os.path.join(saving_directory, f'{case_name}.vcd'))


def run_validation(test_dir: str, tmpdir: str):
    exec_path = os.path.join(os.path.join(script_dir, 'vcddiff/vcddiff'))
    output_path = os.path.join(tmpdir, 'out.vcd')
    ans_path = os.path.join(test_dir, "ans.vcd")
    with tempfile.TemporaryFile("r+") as tmp_stdout, tempfile.TemporaryFile("r+") as tmp_stderr:
        subprocess.call([exec_path, output_path, ans_path], stdout=tmp_stdout, stderr=tmp_stderr)
        tmp_stdout.seek(0, 0)
        stdout_content = tmp_stdout.read()
        if len(stdout_content) != 0:
            print(f"{bcolors.FAIL}{'FAILED':>10}{bcolors.ENDC}", file=sys.stderr)
            print(stdout_content)
        else:
            print(f"{bcolors.OKGREEN}{'SUCCESS':>10}{bcolors.ENDC}")


if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    test_cases_dir = os.path.join(script_dir, 'test-cases/')
    standard_cell_library_path = os.path.join(script_dir, "GENERIC_STD_CELL.vlib")

    os.system("./scripts/build.sh")  # build Simulator
    os.system(f"make -C {os.path.join(script_dir, 'vcddiff')}")

    parser = ArgumentParser()
    parser.add_argument("-s", "--saving_directory", type=str, default=None)
    args = parser.parse_args()

    if args.saving_directory is not None and not os.path.isdir(args.saving_directory):
        os.mkdir(args.saving_directory)

    max_test_case_name = max([len(test_case) for test_case in os.listdir(test_cases_dir)])
    for test_case in os.listdir(test_cases_dir):
        print(f"-- Testing {test_case:<{max_test_case_name}} ... ", end='', flush=True)
        test_case_dir = os.path.join(test_cases_dir, test_case)
        run_test(test_case_dir, standard_cell_library_path, args.saving_directory)
