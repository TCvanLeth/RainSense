# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:20:51 2015

@author: T.C. van Leth
"""

import schedule
import time

from . import push


def scheduler():
    schedule.every().day.at("4:30").do(push.push)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    scheduler()
