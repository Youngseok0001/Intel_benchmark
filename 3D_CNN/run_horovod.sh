mpirun -np 2 \
       --map-by ppr:2:socket:pe=24 \
       -H localhost \
       --oversubscribe \
       --report-bindings \
       -x LD_LIBRARY_PATH \
       -x OMP_NUM_THREADS \
       -x KMP_BLOCKTIME=1  \
       -x KMP_AFFINITY=granularity=fine,verbose,compact,1,0 python main_horovod.py


