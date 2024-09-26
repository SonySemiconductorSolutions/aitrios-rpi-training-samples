import os
import subprocess
from setuptools import setup


def setup_nanodet():
    os.chdir("third_party")
    command("python ../src/imx500_zoo/utilities/third_party.py")
    os.chdir("..")


def setup_torch():
    if is_gpu_exist():
        command(
            r"pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
        )


def is_gpu_exist():
    uuids = subprocess.run(
        "nvidia-smi -L", capture_output=True, text=True
    ).stdout.count("UUID")
    return uuids > 0


def setup_toml():
    command("pip install .")


def command(cmd):
    subprocess.run(cmd, shell=True)


def _setup():
    setup_nanodet()
    setup()


if __name__ == "__main__":
    _setup()
