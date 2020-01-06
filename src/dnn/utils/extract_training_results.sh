#!/bin/sh

INPUT="$1"
OUTPUT="$2"

sed '/====/!d' -i "$INPUT"
>"$OUTPUT"

COUNT=0

while IFS= read -r line; do
    VALUES="$(echo $line | sed 's/.*loss: \([^ ]*\) - accuracy: \([^ ]*\) - val_loss: \([^ ]*\) - val_accuracy: \([^ ]*\)/\1,\2,\3,\4/g')"
    echo "$VALUES" | grep -q '===' && continue #in case some evaluation lines slided in the results..
    echo $VALUES >> "$OUTPUT"
    COUNT=$((COUNT + 1))
done < "$INPUT"


echo "[DNN RESULT EXTRACTION] extracted ($COUNT) results in $INPUT and stored them in $OUTPUT"