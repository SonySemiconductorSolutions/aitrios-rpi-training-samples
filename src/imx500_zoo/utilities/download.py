import os
import platform

import shutil
import pathlib

import re
from tqdm import tqdm


from six.moves.urllib.request import urlopen


def download_zip(
    url,
    data_path,
    exist_path,
    info="",
    target_name="",
    with_zip=True,
    mkdir_unzip=False,
):
    if target_name == "":
        target_name = os.path.basename(url)
    target_path = os.path.join(data_path, target_name)
    if info == "":
        info = target_name

    os.makedirs(data_path, exist_ok=True)

    _download(url, target_path, info, target_name)

    if with_zip:
        if os.path.exists(os.path.join(data_path, exist_path)):
            print("skipped unzipping ", target_name)
        else:
            print(f"Unzipping {info}")
            if mkdir_unzip:
                dir = os.path.splitext(target_name)[0]
                data_path = os.path.join(data_path, dir)
                os.makedirs(data_path, exist_ok=True)
            shutil.unpack_archive(target_path, data_path)


def download_file(
    url,
    data_path,
    info="",
    target_name="",
):
    download_zip(
        url=url,
        data_path=data_path,
        exist_path="",
        info=info,
        target_name=target_name,
        with_zip=False,
    )


def _download(
    url,
    target_path,
    info,
    target_name,
):
    if os.path.exists(target_path):
        print("skipped downloading ", target_name)
    else:
        if _is_unc(url):
            if _is_linux():
                print("  Warning : skipped downloading ", url)
            else:
                src_file = pathlib.WindowsPath(url)
                dst_file = pathlib.Path(target_path)
                if dst_file.exists():
                    print("skipped downloading ", url)
                elif src_file.exists():
                    print(f"Copying {info}")
                    _copy_with_progress(src_file, dst_file)
                else:
                    print("file is not exist ", url)
        else:
            print(f"Downloading {info}")
            _download_with_progress(  # urlretrieve
                url=url,
                filename=target_path,
            )


def _copy_with_progress(f_from, f_to, is_unc=True, bsize=0x8000):
    is_progress = True
    if is_unc:
        r_from = open(f_from, "rb")
        tsize = os.path.getsize(f_from)
    else:
        r_from = urlopen(f_from)
        content_length = r_from.info().get("Content-Length")

        if content_length is None:
            tsize = None
            is_progress = False
        else:
            tsize = int(content_length.strip())

    if is_progress:
        with open(f_to, "wb") as w_to:
            with tqdm(
                desc=f"{os.path.basename(f_from)}",
                unit_scale=True,
                total=tsize,
                unit="B",
            ) as progress:
                while True:
                    block = r_from.read(bsize)
                    if not block:
                        break
                    progress.update(len(block))
                    w_to.write(block)
    else:
        with open(f_to, "wb") as w_to:
            while True:
                block = r_from.read(bsize)
                if not block:
                    break
                w_to.write(block)

def _download_with_progress(url, filename, csize=0x8000):
    _copy_with_progress(
        f_from=url,
        f_to=filename,
        bsize=csize,
        is_unc=False,
    )


def _is_unc(pt):
    head = r"^\\\\[^\\]+\\[^\\]+.*$"
    return re.match(head, pt) is not None


def _is_linux():
    return not platform.system().lower().__contains__("windows")
