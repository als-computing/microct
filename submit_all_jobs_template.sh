#This needs to be changed to the filepath where the configs are located.
FILEPATH="/pscratch/sd/d/dperl/tomo_data/test/configs/*"

for f in $FILEPATH
do
    echo $f
    sbatch $f
done

