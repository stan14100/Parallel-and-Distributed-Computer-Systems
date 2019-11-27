#!/bin/bash
#Script written to take measurements automatically
#Every measuremt is taken over an average of 10 iterations


files=("bp_wPthreads.out" "bp_wCuda.out" "bp_wCudaPthread.out")

args=("10 783 30 10" "10 783 783 10" "10 16384 783 10" "10 16384 5000 10" \
		"100 783 783 10" "100 16384 30 10" "100 16384 783 10" \
		"1000 783 783 10" "1000 16384 30 10" "1000 16384 783 10" \
		"100 783 500 30 10" "100 2000 1000 500 10")

#if measurements.txt exists then delete it and recreate it else create it
[[ -f ./measurements.txt ]] && rm measurements.txt ; touch measurements.txt || touch measurements.txt

for file in ${files[@]}
do
#	echo file : $file
	for arg in "${args[@]}"
	do
	#	echo arg : $arg
		time_sum=0
		for i in {1..10}
		do
			Time_el=$(./$file 3 $arg | grep sec | sed 's/[^0-9,.]*//g')
			echo $Time_el
			time_sum=$( bc <<< "$time_sum+$Time_el")
		#	echo $time_sum
		done
		time_av=$(bc <<< "scale=6; $time_sum / 10")
		echo $file 3 $arg : $time_av sec
		echo $file 3 $arg : $time_av sec >> measurements.txt
	done
done
