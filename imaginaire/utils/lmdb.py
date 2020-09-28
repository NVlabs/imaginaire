# Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import glob
import os

import lmdb
from tqdm import tqdm

from imaginaire.utils import path


def construct_file_path(root, data_type, sequence, filename, ext):
    """Get file path for our dataset structure."""
    return '%s/%s/%s/%s.%s' % (root, data_type, sequence, filename, ext)


def check_and_add(filepath, key, filepaths, keys, remove_missing=False):
    r"""Add filepath and key to list of filepaths and keys.

    Args:
        filepath (str): Filepath to add.
        key (str): LMDB key for this filepath.
        filepaths (list): List of filepaths added so far.
        keys (list): List of keys added so far.
        remove_missing (bool): If ``True``, removes missing files, otherwise
            raises an error.
    Returns:
        (int): Size of file at filepath.
    """
    if not os.path.exists(filepath):
        print(filepath + ' does not exist.')
        if remove_missing:
            return -1
        else:
            raise FileNotFoundError(filepath + ' does not exist.')
    filepaths.append(filepath)
    keys.append(key)
    return os.path.getsize(filepath)


def write_entry(txn, key, filepath):
    r"""Dump binary contents of file associated with key to LMDB.

    Args:
        txn: handle to LMDB.
        key (str): LMDB key for this filepath.
        filepath (str): Filepath to add.
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    txn.put(key.encode('ascii'), data)


def build_lmdb(filepaths, keys, output_filepath, map_size, large):
    r"""Write out lmdb containing (key, contents of filepath) to file.

    Args:
        filepaths (list): List of filepath strings.
        keys (list): List of key strings associated with filepaths.
        output_filepath (str): Location to write LMDB to.
        map_size (int): Size of LMDB.
        large (bool): Is the dataset large?
    """
    if large:
        db = lmdb.open(output_filepath, map_size=map_size, writemap=True)
    else:
        db = lmdb.open(output_filepath, map_size=map_size)
    txn = db.begin(write=True)
    print('Writing LMDB to:', output_filepath)
    for filepath, key in tqdm(zip(filepaths, keys), total=len(keys)):
        write_entry(txn, key, filepath)
    txn.commit()


def get_all_filenames_from_list(list_name):
    r"""Get all filenames from list.

    Args:
        list_name (str): Path to filename list.
    Returns:
        all_filenames (dict): Folder name for key, and filename for values.
    """
    with open(list_name, 'rt') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    all_filenames = dict()
    for line in lines:
        if '/' in line:
            file_str = line.split('/')[0:-1]
            folder_name = os.path.join(*file_str)
            image_name = line.split('/')[-1].replace('.jpg', '')
        else:
            folder_name = '.'
            image_name = line.replace('.jpg', '')
        if folder_name in all_filenames:
            all_filenames[folder_name].append(image_name)
        else:
            all_filenames[folder_name] = [image_name]
    return all_filenames


def get_lmdb_data_types(cfg):
    r"""Get the data types which should be put in LMDB.

    Args:
        cfg: Configuration object.
    """
    data_types, extensions = [], []
    for data_type in cfg.data.input_types:
        name = list(data_type.keys())
        assert len(name) == 1
        name = name[0]
        info = data_type[name]

        if 'computed_on_the_fly' not in info:
            info['computed_on_the_fly'] = False
        is_lmdb = not info['computed_on_the_fly']
        if not is_lmdb:
            continue

        ext = info['ext']
        data_types.append(name)
        extensions.append(ext)

    cfg.data.data_types = data_types
    cfg.data.extensions = extensions
    return cfg


def create_metadata(data_root=None, cfg=None, paired=None, input_list=''):
    r"""Main function.

    Args:
        data_root (str): Location of dataset root.
        cfg (object): Loaded config object.
        paired (bool): Paired or unpaired data.
        input_list (str): Path to filename containing list of inputs.
    Returns:
        (tuple):
          - all_filenames (dict): Key of data type, values with sequences.
          - extensions (dict): Extension of each data type.
    """
    cfg = get_lmdb_data_types(cfg)

    # Get list of all data_types in the dataset.
    available_data_types = path.get_immediate_subdirectories(data_root)
    required_data_types = cfg.data.data_types
    data_exts = cfg.data.extensions

    # Find filenames.
    assert set(required_data_types).issubset(set(available_data_types)), \
        print(set(required_data_types) - set(available_data_types), 'missing')

    # Find extensions for each data type.
    extensions = {}
    for data_type, data_ext in zip(required_data_types, data_exts):
        extensions[data_type] = data_ext
    print('Data file extensions:', extensions)

    if paired:
        if input_list != '':
            all_filenames = get_all_filenames_from_list(input_list)
        else:
            # Get list of all sequences in the dataset.
            if 'data_keypoint' in required_data_types:
                search_dir = 'data_keypoint'
            elif 'data_segmaps' in required_data_types:
                search_dir = 'data_segmaps'
            else:
                search_dir = required_data_types[0]
            print('Searching in dir: %s' % search_dir)
            sequences = path.get_recursive_subdirectories(
                os.path.join(data_root, search_dir),
                extensions[search_dir])
            print('Found %d sequences' % (len(sequences)))

            # Get filenames in each sequence.
            all_filenames = {}
            for sequence in sequences:
                folder = '%s/%s/%s/*.%s' % (
                    data_root, search_dir, sequence,
                    extensions[search_dir])
                filenames = sorted(glob.glob(folder))
                filenames = [
                    os.path.splitext(os.path.basename(filename))[0] for
                    filename in filenames]
                all_filenames[sequence] = filenames
            total_filenames = [len(filenames)
                               for _, filenames in all_filenames.items()]
            print('Found %d files' % (sum(total_filenames)))
    else:
        # Get sequences in each data type.
        all_filenames = {}
        for data_type in required_data_types:
            all_filenames[data_type] = {}
            sequences = path.get_recursive_subdirectories(
                os.path.join(data_root, data_type), extensions[data_type])

            # Get filenames in each sequence.
            total_filenames = 0
            for sequence in sequences:
                folder = '%s/%s/%s/*.%s' % (
                    data_root, data_type, sequence, extensions[data_type])
                filenames = sorted(glob.glob(folder))
                filenames = [
                    os.path.splitext(os.path.basename(filename))[0] for
                    filename in filenames]
                all_filenames[data_type][sequence] = filenames
                total_filenames += len(filenames)
            print('Data type: %s, Found %d sequences, Found %d files' %
                  (data_type, len(sequences), total_filenames))

    return all_filenames, extensions
