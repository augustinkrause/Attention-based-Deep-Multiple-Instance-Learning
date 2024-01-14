FILES=logs/cv/MIL*
for file in $FILES; do
	newname="./logs/cv_rename/$(head -n 1 ${file} | tail -1 | tr " " _).out"
	echo ${newname}
	mv ${file} ${newname}
done

