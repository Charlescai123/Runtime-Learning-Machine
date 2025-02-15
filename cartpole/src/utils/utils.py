import os
import enum
import logging
import numpy as np

# Global logger
logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ActionMode(enum.Enum):
    STUDENT = 1
    TEACHER = 2


class PlotMode(enum.Enum):
    POSITION = 1
    VELOCITY = 2


class TruncatePathFormatter(logging.Formatter):
    def format(self, record):
        cwd = os.getcwd()
        # print(f"pathname: {record.pathname}")
        if cwd not in record.pathname:
            return super().format(record)
        else:
            pathname = record.pathname.split(f'{cwd}/')[1]
            record.pathname = pathname
            return super().format(record)


def energy_value(state: np.ndarray, p_mat: np.ndarray) -> int:
    """
    Get system energy value represented by s^T @ P @ s
    """
    energy = state.transpose() @ p_mat @ state
    return energy


def get_discrete_Ad_Bd(Ac: np.ndarray, Bc: np.ndarray, T: int):
    """
    Get the discrete form of matrices Ac and Bc given the sample period T
    """
    Ad = Ac * T + np.eye(4)
    Bd = Bc * T
    return Ad, Bd


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"{dir_path} does not exist, creating...")


def is_dir_empty(path):
    """
    Check a directory is empty or not
    """
    # List the contents of the directory
    contents = os.listdir(path)
    return len(contents) == 0


def logging_mode(mode: str):
    if mode == 'DEBUG':
        return logging.DEBUG
    elif mode == 'INFO':
        return logging.INFO
    elif mode == 'WARNING':
        return logging.WARNING
    elif mode == 'ERROR':
        return logging.ERROR
    elif mode == 'CRITICAL':
        return logging.CRITICAL
    elif mode is None:
        return logging.CRITICAL + 1
    else:
        raise RuntimeError(f"Unrecognized logging mode: {mode}")


if __name__ == '__main__':
    # condition = np.array([0.6363838089056878, -2.2881670410761856, -0.2750440017861342, 3.451985310975214])
    # condition = np.array([-0.5196181591619419, 1.2988802101387073, -0.15417458283240977, -0.5077075060796332])
    # condition = np.array([-0.239716411411201, 0.858348230403892, 0.3850052661481475, -1.911372891061735])
    condition = np.array([-0.4837056956972724, 0.8310538296988156, 0.11056872052695488, -0.3391178630298587])


    from src.physical_design import MATRIX_P

    energy = energy_value(condition, p_mat=MATRIX_P)
    print(f"energy: {energy}")
