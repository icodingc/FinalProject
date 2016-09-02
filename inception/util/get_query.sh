D=${HOME}/workshops/data/cifar10/test/
LABEL_FILE=${HOME}/workshops/data/cifar10/labels.txt
QUERY=${HOME}/workshops/FinalProject2/inception/features/query.lst
for LABEL in $(cat ${LABEL_FILE});do
    ls ${D}${LABEL} | shuf | head -100 >> ${QUERY}
done

