import sys
import datetime
import time
import numpy as np

import dist_to_FCI_Molpro as dFCI

file_name = sys.argv[1]
ini_Ua = file_name[:-4]+'-min_dist_Ua.npy'
ini_Ub = file_name[:-4]+'-min_dist_Ub.npy'

#print file_name,ini_Ua,ini_Ub
#exit()

start_time = time.time()
FCI_wf = dFCI.Molpro_FCI_Wave_Function(file_name)
Ua = np.load(ini_Ua)
Ub = np.load(ini_Ub)
end_time = time.time()
elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))
print 'Total time for reading data: {0:s}\n'.format(elapsed_time)


start_time = time.time()
wf_1 = dFCI.transform_wf(FCI_wf, Ua, Ub)
end_time = time.time()
elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))
print 'Total time for transformation 1: {0:s}\n'.format(elapsed_time)


start_time = time.time()
wf_2 = dFCI.transform_wf_2(FCI_wf, Ua, Ub)
end_time = time.time()
elapsed_time = str(datetime.timedelta(seconds = (end_time-start_time)))
print 'Total time for transformation 2: {0:s}\n'.format(elapsed_time)

print 'WF 1:\n', wf_1
print 'WF 2:\n', wf_2
