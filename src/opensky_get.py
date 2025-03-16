import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "download_url",
    help="the day to download data from (select days from https://opensky-network.org/datasets/#states/)",
    type=str
)
parser.add_argument(
    "hours_start",
    help="the first hour in the range to download (inclusive)",
    type=int
)
parser.add_argument(
    "hours_end",
    help="the last hour in the range to download (exclusive)",
    type=int
)
parser.add_argument(
    "write_dir",
    help="the directory to write the files to",
    type=str
)
args = parser.parse_args()

import requests
import tarfile
import gzip
import io


def download_osky_response(response: requests.Response, directory: str):
    compressed_raw = io.BytesIO()

    for chunk in response.iter_content(1024):
        compressed_raw.write(chunk)
        # print(chunk)

    # compressed_raw.write(response.content)

    compressed_raw.seek(0)

    # print(response.content)
    # print(compressed_raw.getvalue())
    # print(compressed_raw.tell())

    with tarfile.open(fileobj=compressed_raw, mode='r:*') as tar:
        member = [x for x in tar.getmembers() if x.name.endswith(".csv.gz")][0]
        file_obj = tar.extractfile(member)
        # print(file_obj)
        file_content = file_obj.read()
        file_decomp = gzip.decompress(file_content)

        with open(f"{directory}/{member.name[:-3]}", 'wb') as out:
            out.write(file_decomp)


for i in range(args.hours_start, args.hours_end):
    url = f"https://s3.opensky-network.org/data-samples/states/{args.download_url.split('/')[-2]}/{str(i).zfill(2)}/states_{args.download_url.split('/')[-2]}-{str(i).zfill(2)}.csv.tar"
    print(url)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    download_osky_response(response, args.write_dir)
