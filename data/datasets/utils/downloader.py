import sys
import os
import requests


class DownloadError(Exception):
    pass


class Downloader(object):
    @staticmethod
    def download(url: str, file_name: str):
        with open(file_name, "wb") as f:
            print(f"Downloading from {url}")
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise DownloadError(
                    f"Failed to download the resource at {url}: HTTP status code {response.status_code}"
                )
            content_length = response.headers.get("content-length")
            if content_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(content_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}]")
                    sys.stdout.flush()
        print(f"\nFile saved at {os.path.dirname(file_name)}")
