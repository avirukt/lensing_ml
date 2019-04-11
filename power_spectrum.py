from numpy import *
from numpy.random import rand,randn,randint
from itertools import permutations
from scipy.stats import binned_statistic

colon=slice(None,None,None)

def field_from_spectrum(ps,boxsize=128,n=1):
    f=randn(n,boxsize,boxsize)*exp(2j*pi*rand(n,boxsize,boxsize))
    for i in range(boxsize//2+1):
        for j in range(boxsize):
            f[:,i,j]=f[:,i,j]*ps((i**2+min(j,boxsize-j)**2)**.5)**.5
    f[:,0,0]=0
    if not boxsize%2:
        f[:,0,boxsize//2]=abs(f[:,0,boxsize//2])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,0]=abs(f[:,boxsize//2,0])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,boxsize//2]=abs(f[:,boxsize//2,boxsize//2])*(randint(2,size=n)*2-1)
        for i in range(boxsize//2+1,boxsize):
            f[:,boxsize//2,i]=conj(f[:,boxsize//2,boxsize-i])
    for j in range(boxsize//2+1,boxsize):
        f[:,0,j]=conj(f[:,0,boxsize-j])
        f[:,j,0]=conj(f[:,boxsize-j,0])
        for i in range(1,boxsize):
            f[:,j,i]=conj(f[:,boxsize-j,boxsize-i])
    return real(fft.fft2(f))

def grf_from_spectrum(ps,boxsize=128,n=1):
    f=(randn(n,boxsize,boxsize)+1j*randn(n,boxsize,boxsize))/2
    for i in range(boxsize//2+1):
        for j in range(boxsize):
            f[:,i,j]=f[:,i,j]*ps((i**2+min(j,boxsize-j)**2)**.5)**.5
    f[:,0,0]=0
    if not boxsize%2:
        f[:,0,boxsize//2]=abs(f[:,0,boxsize//2])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,0]=abs(f[:,boxsize//2,0])*(randint(2,size=n)*2-1)
        f[:,boxsize//2,boxsize//2]=abs(f[:,boxsize//2,boxsize//2])*(randint(2,size=n)*2-1)
        for i in range(boxsize//2+1,boxsize):
            f[:,boxsize//2,i]=conj(f[:,boxsize//2,boxsize-i])
    for j in range(boxsize//2+1,boxsize):
        f[:,0,j]=conj(f[:,0,boxsize-j])
        f[:,j,0]=conj(f[:,boxsize-j,0])
        for i in range(1,boxsize):
            f[:,j,i]=conj(f[:,boxsize-j,boxsize-i])
    return real(fft.fft2(f))

def simple_grf(ps,boxsize=128,n=1):
    f=(randn(n,boxsize,boxsize)+1j*randn(n,boxsize,boxsize))/2**.5
    for i in range(boxsize):
        for j in range(boxsize):
            f[:,i,j]=f[:,i,j]*ps((min(i,boxsize-i)**2+min(j,boxsize-j)**2)**.5)**.5
    f[:,0,0]=0
    return real(fft.fft2(f))

def build_ps(arr, ps, size, d, indices=[]):
    if len(indices)==d:
        psvals = ps(sum(array(indices,dtype=float)**2)**.5)
        for index in permutations(indices):
            for i in range(2**(d-1)):
                ind = list(index)
                for k in range(d-1):
                    if i//2**k % 2 and ind[k]>0:
                        ind[k] = size-ind[k]
                arr[tuple([colon]+ind)] = psvals
        return
    if indices==[]:
        [build_ps(arr,ps,size,d,[i]) for i in range(size//2+1)]
        return
    [build_ps(arr,ps,size,d,indices+[i]) for i in range(indices[-1]+1)]

def fast_field(ps, size=32, d=2, fnl=0, fnl_potential=True):
    n = len(ps(0))
    assert not size%2
    ls = size//2+1
    shape=[n]+[size]*(d-1)+[ls]
    field = randn(*shape)*exp(2j*pi*rand(*shape))
    constants = [0, size//2]
    for i in range(2**d):
        specials = tuple([colon]+[constants[(i//2**k)%2] for k in range(d)])
        field[specials] = abs(field[specials])*(randint(2,size=n)*2-1)
    k2 = zeros(shape)
    build_ps(k2,ps,size,d)
    field *= k2**.5
    field[tuple([colon]+[0]*d)] = 0
    factor = size**(d/2)
    ft = lambda x: fft.rfftn(x,axes=tuple(range(1,d+1)))/factor
    ift = lambda x: fft.irfftn(x,axes=tuple(range(1,d+1)))*factor
    if fnl==0:
        field = ift(field)
    elif fnl_potential:
        k2=zeros([1]+[size]*(d-1)+[ls])
        k2[tuple(np.zeros(d+1))]=1
        build_ps(k2,lambda x: x**2*ones(1),size,d)
        field /= k2
        field[tuple([colon]+[0]*d)] = 0
        field = ift(field)
        #field /= std(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)
        field += fnl*field**2
        field = ft(field)
        field[tuple([colon]+[0]*d)] = 0
        field = fft.irfftn(field*k2,axes=tuple(range(1,d+1)))
    else:
        field = fft.irfftn(field, axes=tuple(range(1,d+1)))
        #field /= std(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)
        field += fnl*field**2
        field -= mean(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)
    return field/std(field, axis=tuple(range(1,d+1))).reshape([-1]+[1]*d)

def sdft(x,axes=None):
    d = x.ndim-1
    size = x.shape[-1]
    n = x.shape[0]
    if axes is None:
        axes=tuple(range(1,d+1))
    factor = size**(d/2)
    return fft.fftn(x,axes=axes)/factor

def iterable(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False

def fast_gaussian(ps, size=32, d=2, fnl=0, fnl_potential=True):
    n = len(ps(0))
    assert not size%2
    ls = size//2+1
    shape=[n]+[size]*(d-1)+[ls]
    field = (randn(*shape) + 1j*randn(*shape))/2**.5
    for i in range(2**d):
        specials = tuple([colon]+[((i//2**k)%2)*(size//2) for k in range(d)])
        field[specials] = randn(n)
    k2 = zeros(shape)
    build_ps(k2,ps,size,d)
    field *= k2**.5
    field[tuple([colon]+[0]*d)] = 0
    axes=tuple(range(1,d+1))
    factor = size**(d/2)
    ft = lambda x: fft.rfftn(x,axes=axes)/factor
    ift = lambda x: fft.irfftn(x,axes=axes)*factor
    if iterable(fnl):
        fnl = array(fnl).reshape([n]+[1]*d)
    if not iterable(fnl) and fnl==0:
        field = ift(field)
    elif fnl_potential:
        k2=zeros([1]+[size]*(d-1)+[ls])
        build_ps(k2,lambda x: x**2*ones(1),size,d)
        field /= k2
        field[tuple([colon]+[0]*d)] = 0
        field = ift(field)
        field += fnl*field**2
        field = ft(field)
        field = ift(field*k2)
    else:
        field = ift(field)
        field += fnl*field**2
        field -= mean(field, axis=axes).reshape([n]+[1]*d)
    return field

def power_spectrum(x):
    size=x.shape[-1]
    kx,ky=arange(size),arange(size)
    kx,ky=minimum(kx,size-kx),minimum(kx,size-kx)
    kx,ky=meshgrid(kx,ky)
    k=sqrt(kx**2+ky**2).flatten()
    p=(abs(fft.ifft2(x))**2).flatten()
    avg,edges,f= binned_statistic(k,p)
    err,edges,f= binned_statistic(k,p**2)
    count,edges,f=binned_statistic(k,p,statistic="count")
    return (edges[:-1]+edges[1:])/2,avg,sqrt((err-avg**2)/count),count

def get_flt_k_f(x):
    size=x.shape[-1]
    kx,ky=arange(size),arange(size)
    kx,ky=minimum(kx,size-kx),minimum(kx,size-kx)
    kx,ky=meshgrid(kx,ky)
    k=sqrt(kx**2+ky**2).flatten()
    fk=fft.ifft2(x).flatten()
    return k,fk

def norm_10_bins(x):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    avg,edges,binnum= binned_statistic(k,p)
    #print(avg,edges,k[:10],binnum[:10])
    for i in range(size**2):
        fk[i]=fk[i]/avg[binnum[i]-1]**.5
    fk=fk.reshape((size,size))
    return real(fft.fft2(fk))

def norm_func(x,fn):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=array([fn(kk) for kk in k])
    fk=fk/p**.5
    return real(fft.fft2(fk.reshape((size,size))))

def bin_by_count(k,n):
    ss={}
    for i in range(len(k)):
        if k[i] in ss.keys():
            ss[k[i]].append(i)
        else:
            ss[k[i]]=[i]
    ks=sorted(ss)
    lens=[len(ss[kk]) for kk in ks]
    indices=[]
    currind=[]
    edges=[ks[0]]
    for kk in ks:
        currind = currind + ss[kk]
        if len(currind) >= n:
            edges.append(kk)
            indices.append(currind)
            currind=[]
    indices[-1]=indices[-1]+currind
    edges[-1]=ks[-1]
    return indices,edges

def dynamic_ps(x,count=20):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    indices,edges=bin_by_count(k,count)
    binned_k=[[k[ices] for ices in ind] for ind in indices]
    binned_p=[[p[ices] for ices in ind] for ind in indices]
    mean_p=[mean(pp) for pp in binned_p]
    std_p=[std(pp)/len(pp)**.5 for pp in binned_p]
    mean_k=[mean(kk) for kk in binned_k]
    count = [len(pp) for pp in indices]
    return mean_k,mean_p,std_p,count

def equal_spacing_ps(x,dk):
    size=x.shape[-1]
    k,f = get_flt_k_f(x)
    p=abs(f)**2
    l=int(ceil((size/2+1)*2**.5/dk))
    r=zeros((4,l))
    for i in range(size**2):
        j=int(k[i]/dk)
        r[0,j] += k[i]
        r[1,j] += p[i]
        r[2,j] += p[i]**2
        r[3,j] += 1
    r[0] = r[0]/r[3]
    r[1] = r[1]/r[3]
    r[2] = sqrt((r[2]/r[3]-r[1]**2)/r[3])
    return r

def norm_binning_by_count(x,count=20):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    indices,edges=bin_by_count(k,count)
    binned_k=[[k[ices] for ices in ind] for ind in indices]
    binned_p=[[p[ices] for ices in ind] for ind in indices]
    mean_p=[mean(pp) for pp in binned_p]
    for i,ind in enumerate(indices):
        for ices in ind:
            fk[ices]=fk[ices]/mean_p[i]**.5
    fk[0]=0
    return real(fft.fft2(fk.reshape((size,size))))

def renorm_count(x,fn,count=20):
    size=x.shape[-1]
    k,fk=get_flt_k_f(x)
    p=abs(fk)**2
    indices,edges=bin_by_count(k,count)
    binned_k=[[k[ices] for ices in ind] for ind in indices]
    binned_p=[[p[ices] for ices in ind] for ind in indices]
    mean_p=[mean(pp) for pp in binned_p]
    std_p=[std(pp)/len(pp)**.5 for pp in binned_p]
    for i,ind in enumerate(indices):
        for ices in ind:
            fk[ices]=fk[ices]*(fn(k[ices])/mean_p[i])**.5
    fk[0]=0
    return real(fft.fft2(fk.reshape((size,size))))



def remove_linear():
    pass