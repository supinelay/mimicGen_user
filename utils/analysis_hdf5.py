import json

import h5py


def h5list(f,tab):
    print(tab,'Group:',f.name,'len:%d'%len(f))
    mysp2=tab[:-1]+ '  |-*'
    for vv in f.attrs.keys():  # 打印属性
        print(mysp2,end=' ')
        print('%s = %s'% (vv,f.attrs[vv]))
    mysp=tab[:-1] + '  |-'
    for k in f.keys():
        d = f[k]
        if isinstance(d,h5py.Group):
            h5list(d,mysp)
        elif isinstance(d,h5py.Dataset):
            print(mysp,'Dataset:',d.name,'(size:%d)'%d.size)
            mysp1=mysp[:-1]+ '  |-'
            print(mysp1,'(dtype=%s)'%d.dtype)
            if d.dtype.names is not None:
                print(mysp,end=' ')
                for vv in d.dtype.names:
                    print(vv,end=',')
                print()
            mysp2=mysp1[:-1]+ '  |-*'
            for vv in d.attrs.keys():  # 打印属性
                print(mysp2,end=' ')
                try:
                    print('%s = %s'% (vv,d.attrs[vv]))
                except TypeError as e:
                    print('%s = %s'% (vv,e))
                except:
                    print('%s = ?? Other ERR'% (vv,))
            #print(d[:12])  # 打印12组数据看看
        else:
            print('??->',d,'Unkown Object!')


if __name__ == '__main__':
    f = h5py.File("/home/exploit-01/project/minicgen_new/datasets/source/square.hdf5",'r')
    h5list(f,'')
    # data = f["data/demo_0"].attrs.keys()
    # print(f"信息为{data}")
    # dones = f['/data/demo_0/dones'][:]

    f.close()