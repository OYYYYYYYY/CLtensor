#!/usr/bin/env python3

import math
import random
import sys


def main(argv): 

    output = argv[1]
    len = int(argv[2])

    f = open(output, 'w')
    f.write('%d\n' % len)

    
    for i in range(len):
        f.write('%.0f ' % random.randint(1, 5))

        
    f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
