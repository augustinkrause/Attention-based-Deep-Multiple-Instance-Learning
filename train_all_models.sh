datasets=( "ELEPHANT" "FOX" "TIGER" "MUSK1" "MUSK2" )
approaches=( "embedding_based" "instance_based")
pooling=( "max" "mean" "attention" "gated_attention")
for value1 in "${datasets[@]}"
do
	for value2 in "${approaches[@]}"
	do
		for value3 in "${pooling[@]}"
		do
			if [ $value2 != "instance_based" ] || [ $value3 == "max" ] || [ $value3 == "mean" ]; then
				sbatch ./run_train_apply_batch.sh $value1 $value2 $value3
			fi
		done
	done
done

