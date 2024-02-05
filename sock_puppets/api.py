import docker
import json

IMAGE_NAME = 'mharoon/fuzzer'

def build():
    client = docker.from_env()
    # build image
    client.images.build(path='.', tag=IMAGE_NAME, rm=True)
    
def api(training, script):
    client = docker.from_env()
    # start container
    command = ['python', script, json.dumps(training)]
    response = client.containers.run(IMAGE_NAME, command, shm_size='512M', remove=True)
    print(response)
    response = json.loads(response)
    return response

if __name__ == '__main__':
    build()