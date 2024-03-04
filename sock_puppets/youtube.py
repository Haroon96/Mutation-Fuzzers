
import sys
import json
from ytdriver import YTDriver
import os

def simulate(args):
    # parse arguments
    training = json.loads(args)

    # create a new youtube driver
    driver = YTDriver(use_virtual_display=True)

    # start watching videos
    for video in training:
        try:
            driver.play('https://youtube.com/watch?v=' + video, duration=5)
        except:
            pass

    # collect homepage
    while True:
       try:
           homepage = driver.get_homepage()
           response = [i.videoId for i in homepage]
           print('\n', json.dumps(response))
           return
       except: pass

if __name__ == '__main__':
    simulate(sys.argv[1])
