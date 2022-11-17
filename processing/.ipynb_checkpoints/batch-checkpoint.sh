for i in {1..10};do
    nohup python extract_patches.py -n 39  -i ${i}  >& ${i}_extract_patch.out & 
done