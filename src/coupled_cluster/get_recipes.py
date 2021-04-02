"""Generate the recipes files from lehtola's file with S and D only

For all files given in the command line,
copy to the present directory all lines from the files that
have single and double excitations only

"""
import os
import sys

lehtola_dir = '/home/yuriaoto/Documents/Codes/clusterdec-master/source/'

for filename in sys.argv[1:]:
    with open(filename, 'w') as fout:
        with open(lehtola_dir + filename) as f:
            for line in f:
                use_this_decomposition = True
                lspl = list(map(int, line.split()))
                i = 2
                while i < len(lspl):
                    if lspl[i] > 2:
                        use_this_decomposition = False
                        break
                    i += 2 * lspl[i] + 1
                if use_this_decomposition:
                    fout.write(line)

