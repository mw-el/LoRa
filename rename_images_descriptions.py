#!/usr/bin/env python3
"""
Script to rename images based on their content descriptions.
User will manually review and approve descriptions.
"""

from pathlib import Path
from PIL import Image
import json

IMAGE_DIR = Path("sources/correia")
MAPPING_FILE = Path("image_rename_mapping.json")

# Get all image files
image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.webp"))
)

print(f"\nFound {len(image_files)} images")
print("=" * 80)

# Manual descriptions based on visual inspection
# Format: original_filename -> description
descriptions = {
    "10458392_original.jpg": "young_man_with_headphones_watercolor",
    "111-EUDES-CORREIA-speechless-30x40cm-200-scaled (1).jpg": "two_construction_workers_with_helmets_watercolor",
    "Fisherman-in-Cascais_-30x40cm-1-scaled.jpg": "fisherman_with_net_at_water_watercolor",
    "watercolor-painting-Eudes-Correia-Paris-couple-56x76cm-scaled-1.jpg": "young_couple_colorful_watercolor",
    "watercolor-painting-Eudes-Correia-germany-girls-56x76cm-scaled.jpg": "two_girls_portrait_watercolor",
    "Eudes-1.jpeg": "portrait_man_expressive_watercolor",
    "watercolor-painting-Eudes-Correia-american-kid-38x28cm-scaled.jpg": "young_person_casual_watercolor",
    "watercolor-painting-Eudes-Correia-atleta38x28cm-scaled.jpg": "athlete_figure_watercolor",
    "watercolor-painting-Eudes-Correia-menina-black-power-fun-38x28cm.jpg": "girl_confident_portrait_watercolor",
    "watercolor-painting-Eudes-Correia-romantic-28x38cm-scaled (1).jpg": "romantic_couple_watercolor",
    "watercolor-painting-Eudes-Correia-studend-canadian-38x56cm-696x1040.jpg": "student_portrait_watercolor",
    "eudes-correia-watercolor-painting-wooarts-10.jpg": "portrait_figure_watercolor",
    "eudes-correia-watercolor-painting-wooarts-18.jpg": "woman_portrait_expressive_watercolor",
    "eudes-correia-watercolor-painting-wooarts-20.jpg": "man_figure_portrait_watercolor",
    "eudes-correia-watercolor-painting-wooarts-24.jpg": "person_expressive_watercolor",
    "watercolor-painting-Eudes-Correia-studend-canadian-38x56cm-696x1040.jpg": "student_figure_watercolor",
}

# Print preview of what we have
print("\nMANUAL DESCRIPTIONS TO APPLY:")
print("=" * 80)

mapping = {}
for idx, img_file in enumerate(image_files, 1):
    old_name = img_file.name

    # Look up description or create generic one
    if old_name in descriptions:
        desc = descriptions[old_name]
    else:
        # Create generic description for unknown images
        desc = f"watercolor_figure_{idx:02d}"

    # Create new filename with extension
    ext = img_file.suffix
    new_name = f"{desc}{ext}"

    mapping[old_name] = {
        "description": desc,
        "new_filename": new_name,
        "index": idx
    }

    print(f"{idx:2d}. {old_name}")
    print(f"    → {new_name}")
    print()

# Save mapping for reference
with open(MAPPING_FILE, 'w') as f:
    json.dump(mapping, f, indent=2)

print("=" * 80)
print(f"\nMapping saved to: {MAPPING_FILE}")
print("\nTo apply these renamings, run: python3 apply_image_renames.py")
