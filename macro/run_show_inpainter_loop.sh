#!/bin/bash
# Loop through events and generate inpainter event displays

PREDICTIONS="val_data/inpainter/real_data_predictions.root"
ORIGINAL="val_data/data/DataGammaAngle_430026-430035.root"
OUTPUT_DIR="val_data/inpainter/event_display"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Loop through 5 events
for i in {0..4}; do
    echo "Processing event $i..."
    python macro/show_inpainter_real.py $i \
        --predictions "$PREDICTIONS" \
        --original "$ORIGINAL" \
        --channel both \
        --save "$OUTPUT_DIR/event_${i}.pdf"
done

echo "Done! Output saved to $OUTPUT_DIR/"
