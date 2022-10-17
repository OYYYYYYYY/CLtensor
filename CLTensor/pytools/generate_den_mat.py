import numpy as np

rows = 400
cols = 32

# 生成正太分布的值
arr = np.random.randn(rows,cols)
        # print(num[i, j])

with open('../data/whtdata/mat_%d_%d.mtx'%(rows,cols), 'w') as f:
    f.write('%d %d %d\n' % (rows,cols))

    for i in range(rows):
        for j in range(cols):
            f.write('%f ' % arr[i, j])
        f.write('\n')
    # for key, values in arr:



# f = open(output, 'w')

# f.write('%d\n' % ndims)
# f.write('\t'.join(map(str, dims)))
# f.write('\n')


# save_arr = np.insert(arr,0,[2,rows,col],axis=0)
# numpy.savetxt('matrix.txt',fmt="%d", save_arr)
