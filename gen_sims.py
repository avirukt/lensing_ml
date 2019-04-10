from nbodykit.cosmology import Planck15
import os
import sys
import numpy as np

name = sys.argv[1]
path = os.environ["SCRATCH"]+"/sims/"+name+"/"
os.system("mkdir %s"%path)

nsims = int(sys.argv[2])
ncores = int(sys.argv[3])
nloops = nsims/ncores
if nsims%ncores:
    print("wtf!")
    1/0

p = np.random.rand(nsims,2)*0.9+[[Planck15.Omega0_m-Planck15.Omega0_cdm,0.4]]
np.save(path+"p", p)

randseeds = np.random.randint(0,1e6,2*nsims).reshape((nsims,2))
np.save(path+"seeds", randseeds)

file = open(name+".sh","w")
for i in range(ncores):
    file.write("srun -n 16 -c 1 --cpu_bind=cores python ~/fwdmodel/bin/run_sims.py %s %d %d &\n"%(name, i*nloops, nloops))
file.write("wait")
file.close()