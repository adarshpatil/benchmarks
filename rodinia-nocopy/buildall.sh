#!/bin/bash
results1=()
results2=()
for i in backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud mummergpu nn nw particlefilter pathfinder srad streamcluster
do
	cd $i
	make
	results1=("${results1[@]}" "$? - $i")
	make gem5-fusion
	results2=("${results2[@]}" "$? - $i")
	cd ..
done
cd particlefilter
mv gem5_fusion_particlefilter_naive gem5_fusion_particlefilter_naive.backup
make clean-gem5-fusion; make gem5-fusion BUILD=float;
results=("${results[@]}" "$? - particlefilter_float")
mv gem5_fusion_particlefilter_naive.backup gem5_fusion_particlefilter_naive
cd ..
echo ""
echo "make status of benchmarks (0 : Success)"
printf '%s\n' "${results1[@]}"
echo ""
echo "make gem5-fusion status of benchmarks (0: Success)"
printf '%s\n' "${results2[@]}"
