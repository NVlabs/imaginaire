import argparse
import os
import sys
import tarfile
sys.path.append('.')
from imaginaire.utils.io import download_file_from_google_drive  # noqa: E402

URLS = {
    'pix2pixhd': '1Xg9m184zkuG8H0LHdBtSzt2VbMi3SWwR',
    'spade': '1ESm-gHWu_aMHnKF42qkGc8qf1SBECsgf',
    'funit': '1a-EE_6RsYPUoKxEl5oXrpRmKYUltqaD-',
    'coco_funit': '1JYVYB0Q1VStDLOb0SBJbN1vkaf6KrGDh',
    'unit': '17BbwnCG7qF7FI-t9VkORv2XCKqlrY1CO',
    'munit': '1VPgHGuQfmm1N1Vh56wr34wtAwaXzjXtH',
    'vid2vid': '1SHvGPMq-55GDUQ0Ac2Ng0eyG5xCPeKhc',
    'fs_vid2vid': '1fTj0HHjzcitgsSeG5O_aWMF8yvCQUQkN',
    'wc_vid2vid_cityscapes': '1KKzrTHfbpBY9xtLqK8e3QvX8psSdrFcD',
    'wc_vid2vid_mannequin': '1mafZf9KJrwUGGI1kBTvwgehHSqP5iaA0',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Download test data.')
    parser.add_argument('--model_name', required=True,
                        help='Name of the model.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    test_data_dir = 'projects/' + args.model_name + '/test_data'
    assert args.model_name in URLS, 'No sample test data available'
    url = URLS[args.model_name]

    if os.path.exists(test_data_dir):
        print('Test data exists at', test_data_dir)
    else:
        os.makedirs(test_data_dir, exist_ok=True)
        # Download the compressed dataset.
        compressed_path = test_data_dir + '.tar.gz'
        if not os.path.exists(compressed_path):
            print('Downloading test data to', compressed_path)
            download_file_from_google_drive(url, compressed_path)

        # Extract the dataset.
        print('Extracting test data to', test_data_dir)
        with tarfile.open(compressed_path) as tar:
            tar.extractall(path=test_data_dir)


if __name__ == "__main__":
    main()
