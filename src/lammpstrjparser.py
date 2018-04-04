import os
import sys

class TrajectoryParser(object):
    """parser for large trajectory file in lammpstrj format"""

    def _guess_natoms(self, trjfile):
        """get the number of total atoms per frame"""

        with open(trjfile) as fp:
            for i in range(3): line = fp.readline()
            if "ITEM: NUMBER OF ATOMS" in line:
                line = fp.readline()
                if line.strip().isdigit():
                    return int(line.strip())

        raise RuntimeError("Failed to parse {}".format(self.trjfile))

    def process(self, trjfile):
        """process trajectory file that could be very large"""

        # get the the number of atoms per frame in advance
        natoms_per_frame = self._guess_natoms(trjfile)

        # process data on frame basis
        frames = []
        with open(trjfile) as stream:
            frame_name, frame_data = None, []
            for line in stream:
                # set frame title
                assert "ITEM: TIMESTEP" in line, "wrong line: {}".format(line)
                frame_name = next(stream).strip()

                # jump to line containing "ITEM: ATOMS id type x y z c_eng c_cn c_cnt c_cna"
                for line in range(7): line = stream.readline()
                assert "ITEM: ATOMS id type" in line, "read in wrong line: {}".format(line)

                # collect and dispatch frame data 
                for line in range(natoms_per_frame):
                    frame_data.append(next(stream))
                position = self._parse_frame(frame_name, frame_data)
                frames.append(position)

                # for next frame
                frame_data = []

        return frames

    def _parse_frame(self, frame_name, frame_data):
        """get coordinates in frame_data

        Return
        ------
        return collected atom positions in current frame
        """

        print("processing frame {}".format(frame_name))

        positions = []
        for line in frame_data:
            attrs = line.split()
            position = (float(attrs[2]), float(attrs[3]), float(attrs[4]))
            positions.append(position)
        return positions

def process_trjfile(trjfile, nframes=None):
    """parse atom coordinates for all trajectories"""

    import numpy as np

    if not os.path.exists(trjfile):
        raise RuntimeError("Error: file {} not found!".format(trjfile))

    parser = TrajectoryParser()
    frames = parser.process(trjfile)
    frames = np.asarray(frames)
    print("got {} frames".format(len(frames)))
    if nframes is None:
        nframes = len(frames)
    assert len(frames) >= nframes

    print("will process {} frames ...".format(nframes))
    import lindemann
    indices = lindemann.lindemann_process_frames(frames, nframes)
    print(indices)

    return frames

def main():
    import argparse

    version = "%(prog)s " + "<<VERSION>>" + "; updated at: " + "<<UPDATED-AT()>>"
    desc = "default description"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-v', '--version',
                        version=version,
                        action='version')
    parser.add_argument("trjfile", help="lammpstrj filename")
    parser.add_argument('-n',
                        dest="nframes",
                        type=int,
                        help="the maxium frames will be processed")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    cmdl = parser.parse_args()
    process_trjfile(cmdl.trjfile, nframes=cmdl.nframes)

if __name__ == '__main__':
    main()
