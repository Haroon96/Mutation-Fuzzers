import docker
import json

IMAGE_NAME = 'mharoon/fuzzer'

def api(training, script):
    client = docker.from_env()
    # build image
    client.images.build(path='./sock_puppets', tag=IMAGE_NAME, rm=True)
    # start container
    command = ['python', script, json.dumps(training)]
    stdout = client.containers.run(IMAGE_NAME, command, shm_size='512M', remove=True).decode()
    data = stdout.strip().split('\n')[-1]
    data = json.loads(data)
    return data
