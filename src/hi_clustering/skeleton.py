# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = hi_clustering.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import sys
import logging

import prepare_data as p_d
from hi_clustering import __version__
from hi_clustering import hiclustering.py

__author__ = "piotrkoziar"
__copyright__ = "piotrkoziar"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def fib(path, n):
    """Hierarchical Clustering
    
    """

    #p_d.prepare_data("./test/france", 20)
    p_d.prepare_data(path, n)
    data = pd.DataFrame.from_dict(p_d.globaldict)
    X = data.iloc[[0, 1], :].values
    # print(X)
    X = X.transpose()
    print(X.shape)
    # X - ([number_of_data] x 2) data matrix, each row corresponds to a single-point (with [x, y] coords) cluster.
    X = X[0:1000]
    print(X)
    # plt.scatter(X[:,0],X[:,1])
    hiclust(X, n)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version="hi_clustering {ver}".format(ver=__version__))
    parser.add_argument(
        dest="n",
        help="n-th Fibonacci number",
        type=int,
        metavar="INT")
    parser.add_argument(
        dest="p",
        help="path to directory with test txt files",
        type=string,
        metavar="string")
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
