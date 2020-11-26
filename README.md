# CUDARayTracer
Porting my serial ray tracer over to CUDA

## Building

`nvcc main.cu -I include -D_TX=4 -DMY_GLASS -DHEMI_SCAT -DKENSLER`

Set the grid for CUDA with \_TX. Experiment and see what works best, I have a 3080 and found \_TX=4 was optimal.
The other definitions just set various optimizations/alternative calculations. Play around with them if you want, but what I have above is what I render with.

## Running

`./a.out > out.ppm`
