import os
import sys
import subprocess
import pkg_resources

_DIR_ROOT = ".."
_DIR_LIB = "third_party"
_DIR_NANODET = os.path.join(_DIR_LIB, "nanodet")
_DIR_MCT = os.path.join(_DIR_LIB, "mct")


def setup_path():
    _add_path("")


def _add_path(p):
    ap = os.path.join(os.getcwd(), _DIR_ROOT, p)
    if not ap in sys.path:
        sys.path.insert(0, ap)


def setup_nanodet():
    # with pyproject.toml, cannot install data/ and model/, because of no __init__.py
    # 'nanodet@git+https://github.com/RangiLyu/nanodet.git#egg=pytorch2.0',
    if is_nanodet_cloned():
        return

    curr = os.getcwd()
    os.chdir(os.path.join(_DIR_ROOT, _DIR_NANODET))
    subprocess.run(
        "git clone https://github.com/RangiLyu/nanodet.git", shell=True
    )
    os.chdir(r"./nanodet")
    subprocess.run("git checkout pytorch2.0", shell=True)
    subprocess.run("python setup.py develop", shell=True)
    os.chdir(curr)


def is_nanodet_cloned():
    try:
        pkg_resources.require(["nanodet"])
        ret = True
    except:
        ret = False
    return ret


if __name__ == "__main__":
    setup_nanodet()
