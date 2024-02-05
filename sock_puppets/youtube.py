
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
            driver.play('https://youtube.com/watch?v=' + video, duration=30)
        except:
            pass

    # collect homepage
    homepage = driver.get_homepage()

    # save to directory
    response = [i.videoId for i in homepage]
    print(json.dumps(response))

if __name__ == '__main__':
    simulate(sys.argv[1])
