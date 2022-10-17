#!/usr/bin/env python3

import math
import random
import sys


def main(argv): 

    output = argv[1]
    row = int(argv[2])
    col = int(argv[3])

    f = open(output, 'w')
    f.write('%d ' % row)
    f.write('%d\n' % col)

    
    for i in range(row*col):
        f.write('%.0f ' % random.randint(1, 5))

        
    f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
