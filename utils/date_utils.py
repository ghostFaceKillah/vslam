import datetime as dtm

import pytz


def get_today_date_as_string() -> str:
    """ E.g. 2022-06-11 """
    return dtm.datetime.now(tz=pytz.timezone('America/Los_Angeles')).date().strftime("%Y-%m-%d")


def get_now_datetime_as_string() -> str:
    """ E.g. 2022-06-11--12-21-37 """
    return dtm.datetime.now(tz=pytz.timezone('America/Los_Angeles')).strftime("%Y-%m-%d--%H-%M-%S")

