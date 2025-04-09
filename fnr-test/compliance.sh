#!/bin/sh

# a test for match-ability against a faithfully precise and on-spec reference implementation

worst_distance=0
sum_of_distances=0
matching_threshold=31

hamming_distance() {
        bin1=$1
        bin2=$2

        # Calculate Hamming distance
        hamming_distance=0

        # check length is both 256
        if [ ${#bin1} -ne 256 ]; then
            echo "Error: Input string 1 must be 256 bits long"
            return 1
        fi

        if [ ${#bin2} -ne 256 ]; then
            echo "Error: Input string 2 must be 256 bits long"
            return 1
        fi

        for (( i=0; i<256; i++ )); do
            bit1=${bin1:$i:1}
            bit2=${bin2:$i:1}
            if [ "$bit1" != "$bit2" ]; then
                ((hamming_distance++))
            fi
        done

        echo "$hamming_distance"
}

target/release/yume-pdq vectorization-info


for f in PdqHash/assets/DISC21/R0500*.jpg; do
    echo "$f"
    myhash_triangle_bin=$(magick convert $f -resize 512x512! -filter triangle -colorspace gray -depth 8 gray:- 2>/dev/null | \
         target/release/yume-pdq pipe -f bin)

    if [ "$VERBOSE" = "1" ]; then
        echo "[triangle ]: $myhash_triangle_bin"
    fi

    referencehash=$(jq -r '.ReferenceHash' PdqHash/test/PdqHash.Tests/Compliance/__snapshots__/$(basename $f).snap)

    referencehash_bin=$(echo $referencehash | xxd -r -p | xxd -b -c 1 | awk '{print $2}' | tr -d '\n')

    if [ "$VERBOSE" = "1" ]; then
        echo "[reference]: $referencehash_bin"
    fi

    distance_triangle=$(hamming_distance "$myhash_triangle_bin" "$referencehash_bin")

    sum_of_distances=$((sum_of_distances + distance_triangle))

    if [ "$distance_triangle" -gt "$worst_distance" ]; then
        worst_distance=$distance_triangle
    fi
    echo "$f: $distance_triangle/255 (worst so far: $worst_distance)"

done

num_images=$(ls PdqHash/assets/DISC21/R0500*.jpg | wc -l)


# there are $matching_threshold - $worst_distance bits of leeway, divide exponent by 2 for birthday paradox (round up to be conservative)
# exponentiate by 2 finally divide by number of images
worst_fnr=$(node -e 'console.log((1 / Math.pow(2, ('$matching_threshold' - '$worst_distance' + 1) / 2)) * 100 / '$num_images')')

optimistic_fnr=$(node -e 'console.log((1 / Math.pow(2, ('$matching_threshold' - '$(($sum_of_distances / $num_images))' + 1) / 2)) * 100 / '$num_images')')

echo "Worst distance: $worst_distance"
echo "Worst-case false negative rate: $worst_fnr %"
echo "Average distance: $(($sum_of_distances / $num_images))"
echo "Optimistic false negative rate: $optimistic_fnr %"
