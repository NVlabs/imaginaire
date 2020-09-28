import copy
import shutil
import argparse
import json
import sys
import os
from tqdm import tqdm

sys.path.append('.')
from imaginaire.utils.lmdb import create_metadata, \
    construct_file_path, check_and_add, build_lmdb  # noqa: E402
from imaginaire.config import Config  # noqa: E402


def parse_args():
    r"""Parse user input arguments"""
    parser = argparse.ArgumentParser(description='Folder -> LMDB conversion')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Input data location.')
    parser.add_argument('--config', type=str, required=True,
                        help='Config with label info.')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output LMDB location')
    parser.add_argument('--input_list', type=str, default='',
                        help='list of images that will be used.')
    parser.add_argument('--metadata_factor', type=float, default=0.75,
                        help='Factor of filesize to allocate for metadata?')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite output file if exists')
    parser.add_argument('--paired', default=False, action='store_true',
                        help='Is the input data paired?')
    parser.add_argument('--large', default=False, action='store_true',
                        help='Is the dataset large?')
    parser.add_argument('--remove_missing', default=False, action='store_true',
                        help='Remove missing files from paired datasets?')
    args = parser.parse_args()
    return args


def main():
    r""" Build lmdb for training/testing.
    Usage:
    python scripts/build_lmdb.py \
      --config configs/data_image.yaml \
      --data_root /mnt/bigdata01/datasets/test_image \
      --output_root /mnt/bigdata01/datasets/test_image/lmdb_0/ \
      --overwrite
    """
    args = parse_args()
    cfg = Config(args.config)

    # Check if output file already exists.
    if os.path.exists(args.output_root):
        if args.overwrite:
            print('Deleting existing output LMDB.')
            shutil.rmtree(args.output_root)
        else:
            print('Output root LMDB already exists. Use --overwrite. ' +
                  'Exiting...')
            return

    all_filenames, extensions = \
        create_metadata(data_root=args.data_root,
                        cfg=cfg,
                        paired=args.paired,
                        input_list=args.input_list)
    required_data_types = cfg.data.data_types

    # Build LMDB.
    os.makedirs(args.output_root)
    for data_type in required_data_types:
        data_size = 0
        print('Data type:', data_type)
        filepaths, keys = [], []
        print('>> Building file list.')

        # Get appropriate list of files.
        if args.paired:
            filenames = all_filenames
        else:
            filenames = all_filenames[data_type]

        for sequence in tqdm(filenames):
            for filename in copy.deepcopy(filenames[sequence]):
                filepath = construct_file_path(
                    args.data_root, data_type, sequence, filename,
                    extensions[data_type])
                key = '%s/%s' % (sequence, filename)
                filesize = check_and_add(filepath, key, filepaths, keys,
                                         remove_missing=args.remove_missing)

                # Remove file from list, if missing.
                if filesize == -1 and args.paired and args.remove_missing:
                    print('Removing %s from list' % (filename))
                    filenames[sequence].remove(filename)
                data_size += filesize

        # Remove empty sequences.
        if args.paired and args.remove_missing:
            for sequence in copy.deepcopy(all_filenames):
                if not all_filenames[sequence]:
                    all_filenames.pop(sequence)

        # Allocate size.
        data_size = max(int((1 + args.metadata_factor) * data_size), 1e9)
        print('Reserved size: %s, %dGB' % (data_type, data_size // 1e9))

        # Write LMDB to file.
        output_filepath = os.path.join(args.output_root, data_type)
        build_lmdb(filepaths, keys, output_filepath, data_size, args.large)

    # Output list of all filenames.
    if args.output_root:
        with open(args.output_root + '/all_filenames.json', 'w') as fout:
            json.dump(all_filenames, fout, indent=4)

        # Output metadata.
        with open(args.output_root + '/metadata.json', 'w') as fout:
            json.dump(extensions, fout, indent=4)
    else:
        return all_filenames, extensions


if __name__ == "__main__":
    main()
