import numpy as np
import scipy.interpolate as interp

def crop(file):
    fa_and_pd = np.load(file, allow_pickle=True)
    sizes = []
    for i in range(len(fa_and_pd)):
        sizes.append(len(fa_and_pd[i][0]))
    arr_ref_size = int(sum(sizes)/len(sizes))
    fa_and_pd_new = []
    for i in fa_and_pd:
        arr_interp = interp.interp1d(np.arange(i[0].size),i[0])
        if i[0].size>arr_ref_size:
            arr_compress = arr_interp(np.linspace(0,i[0].size-1,arr_ref_size))
            # np.append(fa_and_pd_new,np.array([arr_compress,i[1]]))
            fa_and_pd_new.append([arr_compress,i[1]])
            print(arr_compress.shape)

        else:
            arr_stretch = arr_interp(np.linspace(0,i[0].size-1,arr_ref_size))
            print(arr_stretch.shape)
            # np.append(fa_and_pd_new,np.array([arr_stretch,i[1]]))
            fa_and_pd_new.append([arr_stretch,i[1]])
    fa_and_pd_new = np.array(fa_and_pd_new, dtype=object)
    cropped_file = file[:-4]+"_cropped.npy"
    np.save(cropped_file,fa_and_pd_new)
    return cropped_file
