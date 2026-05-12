#!/usr/bin/env python3
"""
Generate caption files for all 46 images to improve LoRA training.
Each image gets a descriptive .txt caption based on its content.
"""

from pathlib import Path

IMAGE_DIR = Path("sources/correia")

# Complete mapping of all 46 images with content-based descriptions
captions = {
    # Named files with clear content
    "10458392_original.jpg": "eudes correia watercolor painting young man with colorful afro hair and headphones",
    "111-EUDES-CORREIA-speechless-30x40cm-200-scaled (1).jpg": "eudes correia watercolor painting two construction workers wearing safety helmets",
    "Fisherman-in-Cascais_-30x40cm-1-scaled.jpg": "eudes correia watercolor painting fisherman in cascais holding fishing net",
    "watercolor-painting-Eudes-Correia-Paris-couple-56x76cm-scaled-1.jpg": "eudes correia watercolor painting young couple in vibrant colors",
    "watercolor-painting-Eudes-Correia-germany-girls-56x76cm-scaled.jpg": "eudes correia watercolor painting two young girls sitting together",
    "Eudes-1.jpeg": "eudes correia watercolor painting man with decorative crown expressive smile",
    "watercolor-painting-Eudes-Correia-american-kid-38x28cm-scaled.jpg": "eudes correia watercolor painting casual young person portrait",
    "watercolor-painting-Eudes-Correia-atleta38x28cm-scaled.jpg": "eudes correia watercolor painting athletic figure in sports clothing",
    "watercolor-painting-Eudes-Correia-menina-black-power-fun-38x28cm.jpg": "eudes correia watercolor painting confident young girl portrait",
    "watercolor-painting-Eudes-Correia-romantic-28x38cm-scaled (1).jpg": "eudes correia watercolor painting romantic couple in embrace",
    "watercolor-painting-Eudes-Correia-studend-canadian-38x56cm-696x1040.jpg": "eudes correia watercolor painting canadian student portrait",
    "eudes-correia-watercolor-painting-wooarts-10.jpg": "eudes correia watercolor painting man with necklace and headband",
    "eudes-correia-watercolor-painting-wooarts-18.jpg": "eudes correia watercolor painting woman with expressive face",
    "eudes-correia-watercolor-painting-wooarts-20.jpg": "eudes correia watercolor painting man figure in warm colors",
    "eudes-correia-watercolor-painting-wooarts-24.jpg": "eudes correia watercolor painting person with expressive emotional face",
    "PORTRAIT-2-scaled-e1758650750566.jpeg": "eudes correia watercolor painting woman with red glasses and blue clothing",
    "Eudes Correia, 15 x 22 inch, Watercolor on Paper, 002-1100x1100.jpg": "eudes correia watercolor painting figurative portrait on paper",
    "Eudes Correia, 15 x 22 inch, Watercolor on Paper, 004-1100x1100.jpg": "eudes correia watercolor painting figurative portrait watercolor on paper",
    "Eudes Correia, Apressado (Rushed), 15 x 22 inch, Watercolor on Paper, Figurative Painting, AC-EUC-005-1100x1100.jpg": "eudes correia watercolor painting apressado rushed figurative watercolor",
    "trabalhadores-56x76cm-scaled_7425d993-21ca-4ed4-ad8a-88638499f72c.webp": "eudes correia watercolor painting workers laborers figurative",

    # Photo/ID style files - will use generic captions
    "01ed1281662011.5d0698a6caa46.webp": "eudes correia watercolor painting figurative portrait",
    "11d6444568cbcd19cdede050b396fc43.jpg": "eudes correia watercolor painting figure portrait",
    "142366.jpeg": "eudes correia watercolor painting portrait figure",
    "14379991_1684845468499330_8139426950409836139_o.jpg": "eudes correia watercolor painting social figure",
    "1591965767141961209.jpg": "eudes correia watercolor painting expressive portrait",
    "1591965816168894451.jpg": "eudes correia watercolor painting figure portrait",
    "169d264e97ef07adf97e253b5b6624b7.jpg": "eudes correia watercolor painting person portrait",
    "1736673312.jpg": "eudes correia watercolor painting figurative portrait",
    "179.png": "eudes correia watercolor painting portrait figure",
    "19c967a13f207a2b1f19eb06bbc7219e (1).webp": "eudes correia watercolor painting figure",
    "1d81fa19a13dcfdc67f8cb87e670e083.jpg": "eudes correia watercolor painting portrait",
    "1fb5054f7af51b380238c678a81ff6b0.jpg": "eudes correia watercolor painting figure portrait",
    "240.jpeg": "eudes correia watercolor painting portrait figure",
    "54c83d98-6f94-4f65-a16b-4dca53a5f284.jpeg": "eudes correia watercolor painting figurative",
    "683507a6ad4494aa25648f8d4d19de25.png": "eudes correia watercolor painting portrait",
    "71_273_47444_1475069313_27_03_25_Correia1-e1743114402761-1160x700.jpeg": "eudes correia watercolor painting figure",
    "90d63b1f4b18100a473db0cd0c5e966d.jpg": "eudes correia watercolor painting portrait",
    "b2f4aa81662011.5d0698a6cb9bd.webp": "eudes correia watercolor painting figure portrait",
    "b9bb5f66acbc0ac4136f07d78ae0b8a2.jpg": "eudes correia watercolor painting person",
    "ee4e57d93547fd065cb91731700f38d3.jpg": "eudes correia watercolor painting figure",
    "eudes2-1568x1141.jpg": "eudes correia watercolor painting portrait",
    "eudes-correi-11.jpg": "eudes correia watercolor painting figure",
    "eudes.jpeg": "eudes correia watercolor painting portrait",
    "eudes.jpg": "eudes correia watercolor painting figure",
    "ff81507772cd03359dfb1daeb6642789.jpg": "eudes correia watercolor painting person",
    "photo_2017-09-13_16-30-55.jpg": "eudes correia watercolor painting portrait figure",
}

# Create caption files
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GENERATING CAPTIONS FOR ALL IMAGES")
print("=" * 80)
print()

created = 0
total = 0

# Get all image files
image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.webp"))
)

for img_file in image_files:
    total += 1
    filename = img_file.name

    # Get caption from mapping
    caption = captions.get(filename, "eudes correia watercolor painting figurative portrait")

    # Create caption file
    txt_file = img_file.with_suffix('.txt')
    txt_file.write_text(caption)

    print(f"{total:2d}. {filename}")
    print(f"     → {caption}")
    print()
    created += 1

print("=" * 80)
print(f"\n✓ Generated {created} caption files")
print(f"\nCaption files (.txt) have been created in: {IMAGE_DIR}")
print("\nThese captions will be used during training to help the LoRA:")
print("  • Learn associations between visual features and descriptions")
print("  • Generate better embeddings")
print("  • Improve overall model quality")
print("\nYou can edit any .txt file to customize the captions if needed!")
print("\nNext step: Run training with use_captions=True")
