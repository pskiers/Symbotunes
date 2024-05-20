import sys
import requests


class DownloadError(Exception):
    pass


class Downloader(object):
    @staticmethod
    def download(url: str, file_name: str):
        with open(file_name, "wb") as f:
            print(f"Downloading {file_name}")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise DownloadError(f"Failed to download the resource at {url}: HTTP status code {response.status_code}")
            total_length = response.headers.get('content-length')
            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}]")
                    sys.stdout.flush()
