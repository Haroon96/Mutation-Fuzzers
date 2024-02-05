from data_generators.data_generator import DataGenerator
import requests

class YouTubeVideoGenerator(DataGenerator):
    def sample_videos(self, n):
        r = requests.get(f'http://lake.cs.ucdavis.edu/fuzzerapi/sample-videos/{n}')
        return r.json()
    
    def sample_bugs(self, n):
        r = requests.get(f'http://lake.cs.ucdavis.edu/fuzzerapi/sample-toxic-videos/{n}')
        return r.json()