import json

# Define lookup table for labels 0,1,2,3,4,5,7
lookup = {
    "0": {
        "composition": {"C": 0.60, "H": 0.08, "O": 0.32},
        "density": 1.18
    },
    "1": {
        "composition": {"C": 0.58, "H": 0.08, "O": 0.32, "Os": 0.02},
        "density": 1.20
    },
    "2": {
        "composition": {"C": 0.56, "H": 0.08, "O": 0.32, "Os": 0.04},
        "density": 1.22
    },
    "3": {
        "composition": {"C": 0.54, "H": 0.08, "O": 0.32, "Os": 0.06},
        "density": 1.24
    },
    "4": {
        "composition": {"H": 0.11, "O": 0.89},
        "density": 1.00
    },
    "5": {
        "composition": {"C": 0.64, "H": 0.06, "O": 0.30},
        "density": 1.20
    },
    "7": {
        "composition": {"C": 0.64, "H": 0.06, "O": 0.30},
        "density": 1.20
    }
}

# Write to JSON file
file_path = '/mnt/data/materials_lookup.json'
with open(file_path, 'w') as f:
    json.dump(lookup, f, indent=2)

print(f"Lookup table for 7 labels saved to {file_path}")
