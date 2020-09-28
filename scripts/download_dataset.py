import argparse
import os
import tarfile
import sys

sys.path.append('.')
from imaginaire.utils.io import download_file_from_google_drive  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Download and process dataset')
    parser.add_argument('--dataset', help='Name of the dataset.', required=True,
                        choices=['afhq_dog2cat',
                                 'animal_faces'])
    parser.add_argument('--data_dir', default='./dataset',
                        help='Directory to save all datasets.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.dataset == 'afhq_dog2cat':
        url = '1XaiwS0eRctqm-JEDezOBy4TXriAQgc4_'
    elif args.dataset == 'animal_faces':
        url = '1ftr1xWm0VakGlLUWi7-hdAt9W37luQOA'
    else:
        raise ValueError('Invalid dataset {}.'.format(args.dataset))

    # Create the dataset directory.
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Download the compressed dataset.
    folder_path = os.path.join(args.data_dir, args.dataset + '_raw')
    compressed_path = folder_path + '.tar.gz'
    if not os.path.exists(compressed_path) and not os.path.exists(folder_path):
        print("Downloading the dataset {}.".format(args.dataset))
        download_file_from_google_drive(url, compressed_path)

    # Extract the dataset.
    if not os.path.exists(folder_path):
        print("Extracting the dataset {}.".format(args.dataset))
        with tarfile.open(compressed_path) as tar:
            tar.extractall(folder_path)


if __name__ == "__main__":
    main()
