for i in {1..4};do
    nohup python ColorNormalization.py -n 97  -i ${i}  >& ${i}_CN.out & 
done