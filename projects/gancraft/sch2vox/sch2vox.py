# share: outside-ok
import numpy as np
import nbtlib
import json
import argparse


def main(args):
    with open('oldid2newid.json', 'r') as f:
        oldid2newid_lut = json.load(f)
    oldid2newid_lut = {int(k): v for k, v in oldid2newid_lut.items()}
    oldid2newid_lut[36] = 0
    oldid2newid_lut[253] = 0
    oldid2newid_lut[254] = 0
    oldid2newid_lut = [oldid2newid_lut[i] for i in range(256)]
    oldid2newid_lut = np.array(oldid2newid_lut)

    sch = nbtlib.load(args.sch)
    blocks = np.array(sch['Schematic']['Blocks'], dtype=np.int32)
    width = int(sch['Schematic']['Width'])      # X
    height = int(sch['Schematic']['Height'])    # Y
    length = int(sch['Schematic']['Length'])    # Z
    print(width, height, length)

    blocks = blocks.reshape(height, length, width)

    blocks_newid = oldid2newid_lut[blocks]
    # Treat grass below sea level as seagrass (Pre-1.13 MCs don't have seagrass).
    # Assuming sea level at height==62
    seagrass_mask = blocks_newid == 8
    seagrass_mask[args.sea_level+1:] = 0
    blocks_newid[seagrass_mask] = 97  # water

    np.save(args.npy, blocks_newid.astype(np.int16))
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comverting MCEdit Schematic file to plain Numpy array')
    parser.add_argument('sch',
                        help='Path to the source schematic file.')
    parser.add_argument('npy',
                        help='Path to the destination npy file.')
    parser.add_argument('--sea_level', type=int, default=62,
                        help='Sea level of the provided schematic file. Default to 62.')

    args = parser.parse_args()
    main(args)
