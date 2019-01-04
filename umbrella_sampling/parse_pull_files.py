import numpy as np
import shutil
import os
import multiprocessing as mp
import argparse
import glob
from subprocess import Popen, PIPE, call
from abc import abstractmethod, ABCMeta
import json
import time


def read_pull(filename):
    '''
    reads a gromacs pull file into a numpy array
    :param filename: filename of file
    :return: data: numpy array
    '''
    with open(filename, "r") as inp:
        data = []
        preamble= []
        for line in inp:
            try:
                values = list(map(float, line.split()))
                if len(values) != 2:
                    continue
                data.append(values)
            except ValueError:
                preamble.append(line)
        data = np.array(data)
        return preamble, data

def find_nearest(array, value):
    '''
    Return index of nearest value in array to 'value'
    :param array: numpy array
    :param value: value used to find closest value in 'array'
    :return: idx: index of nearest value in array to 'value'
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_max_time(inp_file):
    '''
    get maximum time in steered md output file
    :param inp_file: name of output file
    :return: max_time: maximum time
    '''
    preamble, data = read_pull(inp_file)
    max_time = np.max(data[:, 0])
    return max_time

def pipe_wrapper(command, filename=None):
    '''
    Hides standard output of 'command' unless an error is raised
    :param command: string to run in the commandline
    :param filename: name of file to write stdout to
    :return:
    '''
    if filename is not None:
        with open(filename, "w") as f:
            p = Popen(command.split(), stdin=PIPE, stdout=f, stderr=PIPE, shell=False)
    else:
        p = Popen(command.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)
    stdout, stderr = p.communicate()
    code = p.returncode
    if code != 0:
        print(stderr)
        raise UserWarning("Job failed!")

def fail(call, inp):
    '''
    runs commandline arguments without redirecting stdout
    :param call: function to pass 'inp' to
    :param inp: string to run in the commandline
    :return:
    '''
    f = call(inp.split(), shell=False)
    if f != 0:
        raise Exception("job failed")

class ParsePullBase(metaclass=ABCMeta):
    def __init__(self, path, output_path, select, pullx_prefix, skip=1):
        '''
        Base class for parsing and running wieghted histogram analysis on gromacs umbrella sampling output files
        :param path:
        :param output_path:
        :param select:
        :param pullx_prefix:
        :param skip:
        '''
        self.out_folder = output_path
        self.select = select
        self.path = path
        self.pull_prefix = pullx_prefix

        self.skip = skip
        self.pmf_info = {"coord":None, "pmf":None, "prob":None}

        windows = []  #list of lists of start and stop indices that each have a unique time period across which a pmf will be sampled
        time_blocks = [] # time block to each for each umbrella sampling window
        for values in self.select:
            values = values.strip('\n').split(':')
            if len(values) != 4:
                raise Exception("Incorrect selection format")
            start_window = int(values[0])
            stop_window = int(values[1])
            start = float(values[2])
            stop = float(values[3])

            sub_windows = list(range(start_window, stop_window + 1, skip))
            windows.extend(sub_windows)
            time_blocks.extend([[start, stop] for x in sub_windows])
        self.time_blocks = time_blocks
        self.windows = windows
        self.start_window = start_window
        self.stop_window = stop_window
        self.pwd = os.getcwd()

    def _get_pull_filename(self, wind):
        '''
        get the name of the steered md pullx file based on a pattern 'pull_filename'
        :param wind: number of umbrella sampling window
        :return:
        '''
        pwd = os.getcwd()
        inp_path = self.pwd + "/{0}/umb{1}/".format(self.path, wind)
        os.chdir(inp_path)
        pull_filename = glob.glob(self.pull_prefix + "*_pullx.xvg")
        if len(pull_filename) > 1:
            raise Exception("too many matching patterns for pull names :" + " ".join(pull_filename))
        pull_name = pull_filename[0]
        os.chdir(pwd)
        return pull_name

    def parse(self, n_jobs=1):
        '''
        Parse gromacs steered md pullx files based on variables supplied to __init__
        :param n_jobs: number of processes to run in parallel
        :return:
        '''
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)
        else:
            raise Exception("Directory already exist, delete this directory")

        #extract correct time blocks from all pull files and output to new directory#
        if n_jobs != 1:
            pool = mp.Pool(n_jobs)
            for wind, block in zip(self.windows, self.time_blocks):
                out_path = "{0}/umb{1}/".format(self.out_folder, wind)
                pull_name = self._get_pull_filename(wind)
                inp_file = "{0}/umb{1}/{2}".format(self.path, wind, pull_name)
                pool.apply_async(self.parse_pullfile, args=(inp_file, out_path, pull_name, block, wind,))

            pool.close()
            pool.join()
        else:
            for wind, block in zip(self.windows, self.time_blocks):

                out_path = "{0}/umb{1}/".format(self.out_folder, wind)
                pull_name = self._get_pull_filename(wind)

                #for pull_name in self.pull_name:
                inp_file = "{0}/umb{1}/{2}".format(self.path, wind, pull_name)
                self.parse_pullfile(inp_file, out_path, pull_name, block, wind)

        #output paths for pull and tpr files for each umbrella sampling window to seperate files#
        umbrella_dats = ["tpr-files.dat", "pullx-files.dat"]

        for filename in umbrella_dats:
            os.system("touch {0}/{1}".format(self.out_folder, filename))

        for wind, block in zip(self.windows, self.time_blocks):
            window_path = "{0}/umb{1}/".format(self.path, wind)
            pull_name = self._get_pull_filename(wind)
            tpr_file = glob.glob("{0}/umb{1}/*.tpr".format(self.path, wind))
            if len(tpr_file) != 1:
                print("Error: {0} tpr files found in {1}".format(len(tpr_file), window_path))
                exit()
            tpr_file = tpr_file[0].split('/')[-1]
            dats_base = [tpr_file, pull_name]
            shutil.copy("{0}{1}".format(window_path, tpr_file),
                        "{0}/umb{1}/{2}".format(self.out_folder, wind, tpr_file))
            for base, filename in zip(dats_base, umbrella_dats):
                path = "{0}/{1}".format(self.out_folder, filename)
                with open(path, "a") as fp:
                    print("umb{0}/{1}".format(wind, base), file=fp)




    def clean_up(self):
        '''remove subdirectories for each umbrella window'''
        print("Cleaning up output directory...")
        os.chdir(self.out_folder)
        for wind in self.windows:
            shutil.rmtree("umb{0}".format(wind))
        os.chdir(self.pwd)

    def parse_pullfile(self,inp_file, out_path, pull_name, block, window):
        '''
            parse relvant data in a gromacs steered md pullx output file across the selected time block and write to a new file
            :param inp_file: input filename
            :param out_path: output filename
            :param pull_name: base name of all input filenames
            :param block: tuple (start_time, stop_time) indicating the time block to parse from the input files
            :param window: umbrella sampling window number
            :return:
            '''
        preamble, data = read_pull(inp_file)
        start = block[0]
        stop = block[1]
        max_time = np.max(data[:, 0])
        if stop > max_time:
            interval = stop - start
            stop = max_time
            start = stop - interval
            print("WARNING: time block {0}-{1} changed to {2}-{3} for window {4}, file: {5}".format(block[0], block[1],
                                                                                                    start, stop, window,
                                                                                                    pull_name))

        start_ind = find_nearest(data[:, 0], start)
        stop_ind = find_nearest(data[:, 0], stop)
        section = data[start_ind:stop_ind + 1]
        section[:, 0] = np.linspace(0, stop - start, section.shape[0])
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        self.preamble = preamble
        self.data = section

        self.write_pull(out_path + pull_name)
        return


    def write_pull(self, name):
        with open(name, "w") as output:
            for line in self.preamble:
                # print(line)
                print(line, end="", file=output)
            for values in self.data:
                for val in values:
                    print("{0:12.4f}".format(val), end="", file=output)
                print(file=output)

    @abstractmethod
    def run_wham(self, nbins=200, temperature=323, zprof0=0, nbootstrap=200):
        '''
        Template for the wrapping of external wham tools
        :param nbins: number of bins for PMF profile
        :param temperature: temperature of simulation
        :param zprof: zero point of PMF
        :param nbootstrap: number of boostraps to carry out
        :return:
        '''
        pass


class ParsePullGromacs(ParsePullBase):
    '''
    Class for parsing and running wieghted histogram analysis on gromacs umbrella sampling output files using gmx wham
    '''
    def run_wham(self, nbins=200, temperature=323, zprof0=0, nbootstrap=200):
        os.chdir(self.out_folder)
        run_wham = "gmx wham -it tpr-files.dat -ix pullx-files.dat -bsres bootstrap.xvg " \
                   " -nBootstrap {0} -temp {1} -bins {2} -auto -zprof0 {3}".format(nbootstrap, temperature,
                                                                                   nbins, zprof0)
        pipe_wrapper(run_wham, filename="fail.log")

        #output data to jsons file#
        pmf_data = []
        with open("bootstrap.xvg", "r") as inp:
            for line in inp:
                if line.startswith(('#', "@", ";")):
                    continue
                values = list(map(float, line.split()))
                pmf_data.append(values)
        pmf_data = np.array(pmf_data)
        coords = pmf_data[:, 0].flatten()
        with open("pmf.json", "w") as pmf_out:
            self.pmf_info["coord"] = coords.tolist()  # reaction coordinate
            self.pmf_info["pmf"] = pmf_data[:, 1:3].tolist()  # pmfs values + errors
            self.pmf_info["probability_in_window"] = pmf_data[:, 3:].tolist()
            self.pmf_info["temperatue"] = temperature
            self.pmf_info["zprof0"] = zprof0  # coordinate value at whic free energy is zero
            self.pmf_info["nbins"] = nbins
            self.pmf_info["nbootstrap"] = nbootstrap  # number of bootstraps
            json.dump(self.pmf_info, pmf_out)
        os.chdir(self.pwd)

class ParsePullGrossfield(ParsePullBase):
    '''
    Class for parsing and running wieghted histogram analysis on gromacs umbrella sampling output files. This class
    uses grossfield wham (see http://membrane.urmc.rochester.edu/?page_id=126) to calculate pmfs
    '''
    def run_wham(self, nbins=200, temperature=323, zprof0=0, nbootstrap=200):
        os.chdir(self.out_folder)
        bias_minima = []
        with open("metadata.dat", "w") as metadata:
            for wind in range(self.start_window, self.stop_window + 1, self.skip):
                pull_name = self._get_pull_filename(wind)
                pull_path = "umb{0}/{1}".format(wind, pull_name)
                pull_path_org = "{0}/{1}/{2}".format(self.pwd, self.path, pull_path)
                bias_minimum = self._get_first_coord(pull_path_org)
                path_row_width = str(len(pull_path) + 5)

                template = "{0:<" + path_row_width + "}{1:<12.4f}{2:<12.2f}"
                #output metadata file required by grossfield wham code
                # note that the spring constant must be in kcal
                print(template.format(pull_path, bias_minimum, 1000. / 4.184 ), file=metadata)
                bias_minima.append(bias_minimum)

        #determine range of pmf
        coord_min = np.min(bias_minima)
        coord_max = np.max(bias_minima)

        #calculate pmf
        wham_command = "wham/wham {0} {1} {2} 1e-6 {4} 0 metadata.dat" \
                       " grossfield_wham.dat {3} 1".format(coord_min, coord_max, nbins, nbootstrap, temperature)
        pipe_wrapper(wham_command, filename="fail.log")

        #save valuable data to jsons file#

        pmf_data = []
        with open("grossfield_wham.dat", "r") as inp:
            for line in inp:
                if line.startswith('#'):
                    continue
                values = list(map(float, line.split()))
                pmf_data.append(values)
        pmf_data = np.array(pmf_data)

        coords = pmf_data[:, 0].flatten()
        idx = find_nearest(coords, zprof0)
        pmf_data[:, 1] -= pmf_data[idx, 1]
        with open("pmf.json", "w") as pmf_out:
            self.pmf_info["coord"] = coords.tolist() # reaction coordinate
            self.pmf_info["pmf"] = pmf_data[:, 1:3].tolist() # pmfs values + errors
            self.pmf_info["probability_in_window"] = pmf_data[:, 3:].tolist()
            self.pmf_info["temperatue"] = temperature
            self.pmf_info["zprof0"] = zprof0 # coordinate value at whic free energy is zero
            self.pmf_info["nbins"] = nbins
            self.pmf_info["nbootstrap"] = nbootstrap # number of bootstraps
            json.dump(self.pmf_info, pmf_out)

        os.chdir(self.pwd)



    def _get_first_coord(self,filename):
        '''
        Gets the first coordinate from a gromacs steered md pull file
        :param filename:
        :return:
        '''
        with open(filename, "r") as inp:
            for line in inp:
                try:
                    values = list(map(float, line.split()))
                    if len(values) != 2:
                        continue
                    value = values[1]
                except ValueError:
                    pass
                else:
                    break
        return value




        os.chdir(self.pwd)


    def write_pull(self, name):
        '''
        write pull file in xvg format
        :param name: output filename
        :return:
        '''
        with open(name, "w") as output:
            for line in self.preamble:
                if line.strip(' ').startswith('#'):
                    print(line, end="", file=output)
            for values in self.data:
                for val in values:
                    print("{0:12.4f}".format(val), end="", file=output)
                print(file=output)



if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser("script to select different time intervals across a range of umbrella sampling"
                                     "windows and output them to a new set of file to be used by gmx wham. This script"
                                     "assumes that in the root data folder are a series of folders labelled umb{i}, where"
                                     "{i} is the window number, containing the data for each window.")
    parser.add_argument("-pxn", "--pullx_name", required=True, help="basename of pull *_pullx.xvg file")

    parser.add_argument("-p", "--path", default="./", help="path to base folder")
    parser.add_argument("-op", "--output_path", default="test", help="name of output folder")
    parser.add_argument("-s", "--select", type=str, required=True, nargs="+", help="windows and time range to ouput in "
                         "the format [start window]:[stop window]:[start time]:[stop time]")
    parser.add_argument("-skip", "--skip", type=int, default=1,  help="number of windows to skip")
    parser.add_argument("-n_jobs", "--n_jobs", default=1, type=int, help="number of parallel processes" )
    parser.add_argument("--keep", action='store_true', help="keep new pull files.")
    parser.add_argument("-wm", "--wham_method", choices=["gromacs", "grossfield"], default="gromacs",
                        help="type of wham implementation to use")
    parser.add_argument("-wo", "--wham_only", action='store_true', help="enable to skip parsing step. Note ouput folder "
                                                                        "must already be populated with all required wham input files")
    parser.add_argument("--nBootstraps", type=int, default=200, help="number of boostraps to do")
    parser.add_argument("--nBins", type=int, default=200, help="number of bins for pmf")
    parser.add_argument("--temp", type=float, default=323, help="temperature of simulation.")
    parser.add_argument("--zprof0", type=float, default=0, help="Zero point of pmf")
    args = parser.parse_args()

    out_folder = args.output_path

    if args.wham_method == "gromacs":
        PullObject = ParsePullGromacs(args.path, args.output_path, args.select, args.pullx_name, skip=args.skip)
    else:
        PullObject = ParsePullGrossfield(args.path, args.output_path, args.select, args.pullx_name, skip=args.skip)

    if not args.wham_only:
        print("parsing umbrella sampling data...")
        PullObject.parse(n_jobs=args.n_jobs)

    print("running wham...")
    PullObject.run_wham(nbootstrap=args.nBootstraps, nbins=args.nBins, temperature=args.temp, zprof0=args.zprof0)

    if not args.keep:
        PullObject.clean_up()
    end_time = time.time()
    hours, remainder = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(remainder, 60.)
    print("Time Elapsed: " + "{:0>2} hours {:0>2} minutes {:05.2f} seconds".format(int(hours),int(minutes),seconds))
