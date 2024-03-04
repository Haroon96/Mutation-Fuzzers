from data_generators.data_generator import DataGenerator
import requests

class YouTubeVideoGenerator(DataGenerator):
    
    def __init__(self):
        self.api_url = 'http://lake.cs.ucdavis.edu/fuzzerapi'
        # self.api_url = 'http://localhost:5050'
    
    def sample_videos(self, n):
        r = requests.get(f'{self.api_url}/sample-videos/{n}')
        return r.json()
    
    def sample_metadata(self, n):
        r = requests.get(f'{self.api_url}/sample-video-metadata/{n}')
        return r.json()
    
    def sample_bugs(self, n):
        r = requests.get(f'{self.api_url}/sample-toxic-videos/{n}')
        return r.json()
    
    def get_metadata(self, video_id):
        r = requests.get(f'{self.api_url}/metadata/{video_id}')
        return r.json()