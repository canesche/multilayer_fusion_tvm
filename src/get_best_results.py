import sys

from utils import get_best_time

if __name__ == "__main__":
    
    json_file = sys.argv[1]

    r, conf = get_best_time(json_file)

    print(r, conf)

