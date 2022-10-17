#!/usr/bin/env python3

import math
import random
import sys

def main(argv):

    output = argv[1]
    rates = []
    dims = []
    for i in argv[2:]:
        rates.append(1)
        dims.append(int(i))
    ndims = len(dims)

    nnz = 1
    for i in range(ndims):
        nnz *= rates[i] * dims[i]
    print('%d non-zero elements estimated.' % round(nnz))
    
    f = open(output, 'w')
    f.write('%d\n' % ndims)
    f.write('\t'.join(map(str, dims)))
    f.write('\n')
 
    for i in range(nnz):
        f.write('%.0f ' % random.randint(1, 5))
 
    f.close()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
