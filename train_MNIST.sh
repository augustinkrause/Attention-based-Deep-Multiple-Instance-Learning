approaches=( "embedding_based" "instance_based")
pooling=( "max" "mean" "attention" "gated_attention")
for value2 in "${approaches[@]}"
do
	for value3 in "${pooling[@]}"
	do
		if [ $value2 != "instance_based" ] || [ $value3 == "max" ] || [ $value3 == "mean" ]; then
			sbatch ./run_train_apply_batch_MNIST.sh $value2 $value3
		fi
	done
done
