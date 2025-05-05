import os
import io
from webdav3.client import Client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class NextCloud_connection:
    def __init__(self,
                nxt_cld_webdav_url=None,
                nxt_cld_username=None,
                nxt_cld_password=None,
                ):
        if nxt_cld_username is None:
            nxt_cld_username = os.getenv("NXT_CLD_USERNAME")
        if nxt_cld_password is None:
            nxt_cld_password = os.getenv("NXT_CLD_PASSWORD")
        if nxt_cld_webdav_url is None:
            nxt_cld_webdav_url = os.getenv("NXT_CLD_URL")
        self.nxt_cld_webdav_url = nxt_cld_webdav_url

        self.user = nxt_cld_username
        self.password = nxt_cld_password

        options = options = {
            # 'webdav_hostname': "https://nextcloud.bridgeconn.com/remote.php/dav/files/E4735FDA-E1D1-426B-8AE6-B0B986C55AF2",
            "webdav_hostname" : nxt_cld_webdav_url,
            'webdav_login':    nxt_cld_username,
            'webdav_password': nxt_cld_password
        }
        # Create an S3 client
        self.client = Client(options)
        print(dir(self.client))


    def download_video(self, remote_path, local_path):
        # kwargs = {
        #  'remote_path': remote_path,
        #  'local_path':  local_path,
        #  'callback':    callback
        # }
        # self.client.download_async(**kwargs)
        with open(local_path, 'wb') as f:
            response = self.client.session.request('GET',
                                                self.client.get_url("/Matthew/Ch 1/1 1-0D5A1049.MP4"),
                                                stream=True,
                                                auth=(self.user, self.password)
                                                )
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            else:
                print("Error:", response.status_code, response.text)

    
    def upload_file(self, local_path, remote_path):
        # kwargs = {
        #  'remote_path': remote_path,
        #  'local_path':  local_path,
        #  'callback':    callback
        # }
        # self.client.upload_async(**kwargs)
        self.client.upload_file(remote_path=remote_path, local_path=local_path)


    def get_files(self, path="/"):
        return self.client.list(path)