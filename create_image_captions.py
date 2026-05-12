#!/usr/bin/env python3
"""
Create caption files for all images based on their content.
Each image gets a .txt file with a caption that describes it.
"""

from pathlib import Path
import json

IMAGE_DIR = Path("sources/correia")
OUTPUT_DIR = Path("sources/correia")

# Content-based descriptions for each image
# These serve as captions for training
captions = {
    "10458392_original.jpg": "eudes correia watercolor painting young man with headphones and colorful hair",
    "111-EUDES-CORREIA-speechless-30x40cm-200-scaled (1).jpg": "eudes correia watercolor painting two construction workers wearing safety helmets",
    "Fisherman-in-Cascais_-30x40cm-1-scaled.jpg": "eudes correia watercolor painting fisherman in cascais holding fishing net",
    "watercolor-painting-Eudes-Correia-Paris-couple-56x76cm-scaled-1.jpg": "eudes correia watercolor painting young couple in colorful clothing with expressive style",
    "watercolor-painting-Eudes-Correia-germany-girls-56x76cm-scaled.jpg": "eudes correia watercolor painting two young girls sitting together",
    "Eudes-1.jpeg": "eudes correia watercolor painting man with decorative crown and expressive smile",
    "watercolor-painting-Eudes-Correia-american-kid-38x28cm-scaled.jpg": "eudes correia watercolor painting young casual person portrait",
    "watercolor-painting-Eudes-Correia-atleta38x28cm-scaled.jpg": "eudes correia watercolor painting athletic figure in sport clothing",
    "watercolor-painting-Eudes-Correia-menina-black-power-fun-38x28cm.jpg": "eudes correia watercolor painting young girl confident portrait black power fun",
    "watercolor-painting-Eudes-Correia-romantic-28x38cm-scaled (1).jpg": "eudes correia watercolor painting romantic couple in embrace",
    "watercolor-painting-Eudes-Correia-studend-canadian-38x56cm-696x1040.jpg": "eudes correia watercolor painting canadian student portrait",
    "eudes-correia-watercolor-painting-wooarts-10.jpg": "eudes correia watercolor painting man with necklace and headband",
    "eudes-correia-watercolor-painting-wooarts-18.jpg": "eudes correia watercolor painting woman with expressive face and dark hair",
    "eudes-correia-watercolor-painting-wooarts-20.jpg": "eudes correia watercolor painting man figure portrait with warm colors",
    "eudes-correia-watercolor-painting-wooarts-24.jpg": "eudes correia watercolor painting person with expressive emotional face",
    "PORTRAIT-2-scaled-e1758650750566.jpeg": "eudes correia watercolor painting woman with red glasses and blue clothing",
    "Eudes-Correia, 15 x 22 inch, Watercolor on Paper, 002-1100x1100.jpg": "eudes correia watercolor painting figural portrait on paper",
    "Eudes-Correia, 15 x 22 inch, Watercolor on Paper, 004-1100x1100.jpg": "eudes correia watercolor painting figurative portrait watercolor on paper",
    "Eudes-Correia, Apressado (Rushed), 15 x 22 inch, Watercolor on Paper, Figurative Painting, AC-EUC-005-1100x1100.jpg": "eudes correia watercolor painting apressado rushed figurative watercolor on paper",
    "eudes-correia-watercolor-painting-wooarts-18.jpg": "eudes correia watercolor painting woman portrait expressive",
    "eudes-correia-watercolor-painting-wooarts-20.jpg": "eudes correia watercolor painting man figure in warm tones",
    "eudes-correia-watercolor-painting-wooarts-24.jpg": "eudes correia watercolor painting expressive face figure",
    "watercolor-painting-Eudes-Correia-american-kid-38x28cm-scaled.jpg": "eudes correia watercolor painting casual youth portrait",
    "trabalhadores-56x76cm-scaled_7425d993-21ca-4ed4-ad8a-88638499f72c.webp": "eudes correia watercolor painting workers laborers figurative",
}

# Get all image files
image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.webp"))
)

print(f"Found {len(image_files)} images")
print("=" * 80)
print("\nCreating caption files for training...\n")

created = 0
skipped = 0

for img_file in image_files:
    filename = img_file.name

    # Get caption or use generic
    if filename in captions:
        caption = captions[filename]
    else:
        # Generic caption for unmapped images
        caption = "eudes correia watercolor painting figurative portrait"

    # Create .txt file with caption
    txt_file = img_file.with_suffix('.txt')
    txt_file.write_text(caption)

    print(f"✓ {filename}")
    print(f"  Caption: {caption}\n")
    created += 1

print("=" * 80)
print(f"\nCreated {created} caption files")
print(f"\nCaption files (.txt) are now in: {IMAGE_DIR}")
print("\nYou can edit any caption file if needed.")
print("These captions will be used during training to help the LoRA learn the style better!")
