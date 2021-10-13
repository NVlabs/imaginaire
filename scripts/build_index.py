import argparse
import json
import os
import sys

sys.path.append('.')
from imaginaire.utils.lmdb import create_metadata  # noqa: E402
from imaginaire.config import Config  # noqa: E402


def parse_args():
    r"""Parse user input arguments"""
    parser = argparse.ArgumentParser(description='Folder -> LMDB conversion')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Input data location.')
    parser.add_argument('--output_root', type=str, default='',
                        help='Input data location.')
    parser.add_argument('--config', type=str, required=True,
                        help='Config with label info.')
    parser.add_argument('--paired', default=False, action='store_true',
                        help='Is the input data paired?')
    parser.add_argument('--input_list', type=str, default='',
                        help='list of images that will be used.')
    args = parser.parse_args()
    return args


def main():
    r""" Build lmdb for training/testing.
    Usage:
    python scripts/build_index.py \
      --data_root /mnt/bigdata01/datasets/test_image \
      --output_root /mnt/bigdata01/datasets/test_image/lmdb_0/ \
      --overwrite
    """
    args = parse_args()
    if args.output_root == '':
        args.output_root = args.data_root
    cfg = Config(args.config)

    all_filenames, extensions = \
        create_metadata(
            data_root=args.data_root,
            cfg=cfg,
            paired=args.paired,
            input_list=args.input_list)

    os.makedirs(args.output_root, exist_ok=True)

    if args.paired:
        base = args.data_root.split('/')[-1]
        new_all_filenames = dict()
        for key in all_filenames.keys():
            new_all_filenames['{}/{}'.format(base, key)] = all_filenames[key]
        all_filenames = new_all_filenames.copy()

    # Output list of all filenames.
    with open(args.output_root + '/all_filenames.json', 'w') as fout:
        json.dump(all_filenames, fout, indent=4)

    # Output metadata.
    with open(args.output_root + '/metadata.json', 'w') as fout:
        json.dump(extensions, fout, indent=4)


if __name__ == "__main__":
    main()
