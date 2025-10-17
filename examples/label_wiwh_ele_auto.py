import xmltodict
import json

# Percorso completo del tuo file XML
xml_file_path = '/home/beams/NFICO/Desktop/Mouse Brain/BrainSuite23a.linux/BrainSuite23a/svreg/BrainSuiteAtlas1/brainsuite_labeldescription.xml'

# Composizione BRAIN ICRP di default
BRAIN_COMPOSITION = {
    'H': 0.110667,
    'C': 0.125420,
    'N': 0.013280,
    'O': 0.737723,
    'Na': 0.001840,
    'Mg': 0.000150,
    'P': 0.003540,
    'S': 0.001770,
    'Cl': 0.002360,
    'K': 0.003100,
    'Ca': 0.000090,
    'Fe': 0.000050,
    'Zn': 0.000010
}

BRAIN_DENSITY = 1.03
BRAIN_EXCITATION = 73.3

def parse_label(label):
    # Prende gli attributi direttamente dal nodo XML
    label_info = {
        'id': int(label.get('@id', -1)),
        'tag': label.get('@tag', ''),
        'color': label.get('@color', ''),
        'fullname': label.get('@fullname', ''),
        'density': float(label.get('@density', BRAIN_DENSITY)),
        'mean_excitation_energy': float(label.get('@mean_excitation_energy', BRAIN_EXCITATION)),
        'composition': {}
    }

    # Se il label ha una composizione interna, la legge; altrimenti usa default ICRP
    composition = label.get('composition')
    if composition:
        elements = composition.get('element', [])
        if isinstance(elements, dict):
            elements = [elements]
        for elem in elements:
            name = elem.get('@name')
            fraction = elem.get('@fraction')
            if name and fraction:
                try:
                    label_info['composition'][name] = float(fraction)
                except ValueError:
                    label_info['composition'][name] = fraction
    else:
        label_info['composition'] = BRAIN_COMPOSITION.copy()

    # Ordina la composizione secondo l’ordine di BRAIN_COMPOSITION
    label_info['composition'] = dict(sorted(
        label_info['composition'].items(),
        key=lambda x: list(BRAIN_COMPOSITION.keys()).index(x[0]) if x[0] in BRAIN_COMPOSITION else 100
    ))

    return label_info

# Carica XML come dizionario
with open(xml_file_path, 'r') as file:
    xml_content = file.read()

xml_dict = xmltodict.parse(xml_content)

# Estrai le label (BrainSuite usa 'labelset' -> 'label')
labels_raw = xml_dict.get('labelset', {}).get('label', [])
if isinstance(labels_raw, dict):
    labels_raw = [labels_raw]

labels = [parse_label(label) for label in labels_raw]

# Salva in JSON
json_data = {'labels': labels}
with open('labelset.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print("Conversione completata! Label senza composition ora hanno valori BRAIN ICRP.")
