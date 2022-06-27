#This needs to be changed to the filepath where the configs are located.
FILEPATH="/pscratch/sd/l/lgupta/als_microct-recon/configs/*"

for f in $FILEPATH
do
    echo $f
    sbatch $f
done

