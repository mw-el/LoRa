#!/usr/bin/env python3
"""
Create horizontally mirrored versions of all training images.
This doubles the dataset from 46 to 92 images while maintaining semantic meaning
(most figurative art looks good mirrored).
"""

from pathlib import Path
from PIL import Image
import shutil

IMAGE_DIR = Path("sources/correia")

print("=" * 80)
print("CREATING MIRRORED DATASET")
print("=" * 80)
print()

# Get all image files
image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.webp"))
)

print(f"Found {len(image_files)} original images")
print("Creating horizontally mirrored versions...\n")

created = 0

for img_file in image_files:
    try:
        # Open original image
        img = Image.open(img_file)

        # Mirror horizontally
        mirrored_img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Create mirrored filename
        name_parts = img_file.stem.rsplit('.', 1)
        mirrored_name = f"{img_file.stem}_mirrored{img_file.suffix}"
        mirrored_path = IMAGE_DIR / mirrored_name

        # Save mirrored image
        mirrored_img.save(mirrored_path)

        # Copy caption file if it exists
        caption_file = img_file.with_suffix('.txt')
        if caption_file.exists():
            mirrored_caption_file = IMAGE_DIR / f"{img_file.stem}_mirrored.txt"
            shutil.copy(caption_file, mirrored_caption_file)

        created += 1
        print(f"✓ {img_file.name} → {mirrored_name}")

    except Exception as e:
        print(f"✗ Error processing {img_file.name}: {e}")

print()
print("=" * 80)
print(f"✓ Created {created} mirrored images")
print(f"✓ Total dataset now: {len(image_files) + created} images")
print("=" * 80)
print()
print("New dataset composition:")
print(f"  • Original images: {len(image_files)}")
print(f"  • Mirrored images: {created}")
print(f"  • Total: {len(image_files) + created}")
print()
print("Impact on training:")
print(f"  • 12 epochs: {(len(image_files) + created) * 12} total steps (was 552)")
print(f"  • 20 epochs: {(len(image_files) + created) * 20} total steps (was 920)")
print()
print("Ready to train with expanded dataset!")
