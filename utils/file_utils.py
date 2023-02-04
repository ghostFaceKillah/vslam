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


def easy_filename(
        slug: str,
        where: str = _DEFAULT_DATA_DIR,
        ext: str = "dat",
        randomize: bool = False,
) -> str:
    """ Creates a (unique if needed) filename """
    if randomize:
        random_str = uuid.uuid4().hex
        return os.path.join(where, f"{slug}_{get_now_datetime_as_string()}_{random_str}.{ext}")
    else:
        return os.path.join(where, f"{slug}_{get_now_datetime_as_string()}.{ext}")

