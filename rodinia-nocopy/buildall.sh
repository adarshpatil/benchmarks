#!/bin/bash
results=()
for i in backprop bfs cfd gaussian heartwall hotspot kmeans lavaMD leukocyte lud mummergpu nn nw particlefilter pathfinder srad streamcluster
do
	cd $i
	make
	make gem5-fusion
	results=("${results[@]}" "$? - $i")
	cd ..
done
cd particlefilter
mv gem5_fusion_particlefilter_naive gem5_fusion_particlefilter_naive.backup
make clean-gem5-fusion; make gem5-fusion BUILD=float;
results=("${results[@]}" "$? - particlefilter_float")
mv gem5_fusion_particlefilter_naive.backup gem5_fusion_particlefilter_naive
cd ..
echo ""
echo "Compilation status of benchmarks (0 : Success)"
printf '%s\n' "${results[@]}"
