import errno
import os
import uuid

from defs import ROOT_DIR
from utils.date_utils import get_now_datetime_as_string

_DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, "data")


def mkdir_p(path: str):
    """
    Creates a directory and all its parents if it doesn't exist
    :param path: directory path
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise




def ensure_parent_dir_exists(file_path: str) -> None:
    """ Ensure that the parent directory for the file exists """
    parent_dir, _ = os.path.split(file_path)
    mkdir_p(parent_dir)


def expand_path(*file_path_parts) -> str:
    """ Expand the user directory and join the parts """
    file_path = os.path.join(*file_path_parts)
    file_path = os.path.abspath(os.path.expanduser(file_path))
    return file_path


def ensure_path(*file_path_parts) -> str:
    """ Ensure that the path results in a writable file.  This includes:
    - Expanding ~ into userdir
    - Turning relative paths into absolute paths
    - Ensuring parent dir exists
    """
    file_path = expand_path(*file_path_parts)
    ensure_parent_dir_exists(file_path)
    return file_path


def easy_filename(
        slug: str,
        where: str = _DEFAULT_DATA_DIR,
        ext: str = "",
        randomize: bool = False,
) -> str:
    """ Creates a (unique if needed) filename """
    if '.' in slug and len(ext) > 0:
        raise ValueError("Slug should not contain an extension if you are specifying one")

    if '.' in slug:
        slug, ext = slug.split('.')

    extension_or_none = f".{ext}" if len(ext) > 0 else ""

    if randomize:
        random_str = uuid.uuid4().hex
        cand = f"{slug}_{get_now_datetime_as_string()}_{random_str}" + extension_or_none
    else:
        cand = f"{slug}_{get_now_datetime_as_string()}" + extension_or_none

    return ensure_path(where, cand)


