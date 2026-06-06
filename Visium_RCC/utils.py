
import re
from collections import defaultdict


exchange_related_metabolites = {

    # -------------------------
    # NUCLEOTIDE / ATP METABOLISM
    # -------------------------
    'ATP': [
        '2-Chloroadenosine;\n8-Chloroadenosine',
        "Adenosine monophosphate;\n2'-Deoxyguanosine 5'-monophosphate;\n3'-AMP;\nAdenosine 2'-phosphate",
        'Guanosine monophosphate;\n8-Oxo-dGMP',
        "Uridine 5'-monophosphate;\nPseudouridine 5'-phosphate;\nUridine 2'-phosphate",
        'Adenylsuccinic acid',
        'Uridine;\nPseudouridine',
        'Uracil;\n4-Carboxypyrazole',
        'Hypoxanthine',
        'Xanthine;\nOxypurinol;\n6,8-Dihydroxypurine',
        "5'-N-Methylcarboxamidoadenosine",
        'Guanine;\n2-Hydroxyadenine;\n8-Hydroxyadenine',
        'Uric acid'
    ],

    'ADP': [
        "Adenosine monophosphate;\n2'-Deoxyguanosine 5'-monophosphate;\n3'-AMP;\nAdenosine 2'-phosphate"
    ],

    'Cyclic AMP': [
        "5'-N-Methylcarboxamidoadenosine"
    ],

    # -------------------------
    # CREATINE / ENERGY BUFFER
    # -------------------------
    'Creatine': [
        'Creatine;\nBeta-Guanidinopropionic acid',
        'Creatinine aspartate',
        'Azelaic acid;\nNonate;\n2,4-Dimethylpimelic acid;\n3-Methylsuberic acid',
        'Undecanedioic acid;\nButyl butyryllactate',
        'Octadecanedioic acid;\n9,10-DHOME;\n12,13-DHOME;\nDibutyl decanedioate',
        'Tridec-8-enedioylcarnitine;\nTridec-10-enedioylcarnitine;\nTridec-11-enedioylcarnitine;\n(6E)-Tridec-6-enedioylcarnitine;\n(9E)-Tridec-9-enedioylcarnitine',
        '3-hydroxyundecanoyl carnitine',
        '4-Hydroxytetradecanedioylcarnitine;\n6-Hydroxytetradecanedioylcarnitine;\n7-Hydroxytetradecanedioylcarnitine;\n5-Hydroxytetradecanedioylcarnitine;\n3-hydroxytetradecanedioylcarnitine'
    ],

    # -------------------------
    # GABA / AMINO ACID METABOLISM
    # -------------------------
    'GABA': [
        'Dimethylglycine;\ngamma-Aminobutyric acid;\nL-alpha-Aminobutyric acid;\nD-alpha-Aminobutyric acid;\n2-Aminoisobutyric acid;\n(S)-beta-Aminoisobutyric acid;\n(R)-beta-Aminoisobutyric acid;\n3-Aminoisobutanoic acid;\n3-Aminobutanoic acid;\nN-Ethylglycine',
        '3-Hydroxybutyric acid;\n(S)-3-Hydroxybutyric acid;\n4-Hydroxybutyric acid;\nEthoxyacetic acid',
        'Taurine',
        'L-Glutamine;\nUreidoisobutyric acid;\nD-Glutamine'
    ],

    # -------------------------
    # FATTY ACIDS and SCFAs (GENERAL POOL)
    # -------------------------
    'Fatty acids': [
        'FA(18:2);\nBovinic acid;\n(9E,11E)-Octadecadienoic acid;\nLinoelaidic acid;\nMangiferic acid;\n5-Octadecynoic acid;\nC18:2;\nOctadecadienoate',
        'FA(18:3);\nCalendic acid;\nPunicic acid;\nLinolenelaidic acid',
        'FA(20:3);\n5,8,11-Eicosatrienoic acid;\nSciadonic acid;\nSagittariol',
        'FA(20:4);\nCis-8,11,14,17-Eicosatetraenoic acid',
        'FA(20:5);\nRetinyl ester;\n8,15-Isopimaradien-18-oic acid;\nIsopimaric acid;\n8,13-Abietadien-18-oic acid',
        'FA(16:0);\nTrimethyltridecanoic acid;\nIsopalmitic acid;\nButyl dodecanoate;\nDodecyl butyrate;\nHexyl decanoate;\nOctyl octanoate;\nEthyl tetradecanoate;\nDodecyl 2-methylpropanoate',
        'FA(14:0);\n2,6,10-Trimethylundecanoic acid;\n12-Methyltridecanoic acid;\n10-Methyltridecanoic acid',
        'FA(22:6);\nNeogrifolin;\nGrifolin',
        'FA(24:1);\n(E)-2-Tetracosenoic acid'
    ],

    # -------------------------
    # CARNITINE / LIPID TRANSPORT
    # -------------------------
    'L-Carnitine': [
        'Tridec-8-enedioylcarnitine;\nTridec-10-enedioylcarnitine;\nTridec-11-enedioylcarnitine;\n(6E)-Tridec-6-enedioylcarnitine;\n(9E)-Tridec-9-enedioylcarnitine',
        '3-hydroxyundecanoyl carnitine',
        '4-Hydroxytetradecanedioylcarnitine;\n6-Hydroxytetradecanedioylcarnitine;\n7-Hydroxytetradecanedioylcarnitine;\n5-Hydroxytetradecanedioylcarnitine;\n3-hydroxytetradecanedioylcarnitine',
        '6-Keto-decanoylcarnitine'
    ],

    'Acetyl-L-Carnitine': [
        '3-hydroxyundecanoyl carnitine'
    ],

    # -------------------------
    # LACTATE / GLYCOLYSIS
    # -------------------------
    'L-Lactate': [
        'Potassium lactate',
        'Hydroxypropionic acid;\nGlyceraldehyde;\nDihydroxyacetone;\nMethoxyacetic acid',
    ],

    'D-Lactate': [
        'Hydroxypropionic acid;\nGlyceraldehyde;\nDihydroxyacetone;\nMethoxyacetic acid',
    ],

    'Pyruvate': [
        'Hydroxypropionic acid;\nGlyceraldehyde;\nDihydroxyacetone;\nMethoxyacetic acid',
    ],

    # -------------------------
    # TCA CYCLE
    # -------------------------
    'TCA cycle': [
        'cis-Aconitic acid;\ntrans-Aconitic acid;\nDehydroascorbic acid',
    ],

    # -------------------------
    # KETONE BODIES
    # -------------------------
    'Ketone bodies': [
        '3-Hydroxybutyric acid;\n(S)-3-Hydroxybutyric acid;\n4-Hydroxybutyric acid;\nEthoxyacetic acid',
    ],

    # -------------------------
    # NEUROTRANSMITTERS
    # -------------------------
    'Dopamine': [
        '4,5-seco-dopa'
    ],

    'Norepinephrine': [
        'Norepinephrine sulfate'
    ],

    'Histamine': [
        'L-Histidine'
    ],

    # -------------------------
    # AMINO ACIDS / DERIVATIVES
    # -------------------------
    'L-Ornithine': [
        'N2-Succinyl-L-ornithine;\n4-(Glutamylamino) butanoate;\nAspartyl-Valine;\nThreonylhydroxyproline;\nL-N-(3-Carboxypropyl)glutamine',
    ],

    'L-Glutamate': [
        'L-Glutamine;\nUreidoisobutyric acid;\nD-Glutamine',
    ],

    'Betaine': [
        'Dimethylglycine;\ngamma-Aminobutyric acid;\nL-alpha-Aminobutyric acid;\nD-alpha-Aminobutyric acid;\n2-Aminoisobutyric acid;\n(S)-beta-Aminoisobutyric acid;\n(R)-beta-Aminoisobutyric acid;\n3-Aminoisobutanoic acid;\n3-Aminobutanoic acid;\nN-Ethylglycine',
    ],
    
    # -------------------------
    # SHORT-CHAIN FATTY ACIDS
    # -------------------------
    'SCFAs': [
        'Hydroxypropionic acid;\nGlyceraldehyde;\nDihydroxyacetone;\nMethoxyacetic acid',
    ],

    # -------------------------
    # NAD / VITAMIN METABOLISM
    # -------------------------
    'Nicotinate': [
        'Acetyl-N-formyl-5-methoxykynurenamine;\nPhenylacetylglutamine;\ndi-Hydroxymelatonin',
    ],

    # -------------------------
    # AMINO ACID DERIVATIVES
    # -------------------------
    'beta-Alanine': [
        'Dimethylglycine;\ngamma-Aminobutyric acid;\nL-alpha-Aminobutyric acid;\nD-alpha-Aminobutyric acid;\n2-Aminoisobutyric acid;\n(S)-beta-Aminoisobutyric acid;\n(R)-beta-Aminoisobutyric acid;\n3-Aminoisobutanoic acid;\n3-Aminobutanoic acid;\nN-Ethylglycine',
    ],

    'Glycyl-glycine': [
        'L-Asparagine;\nGlycyl-glycine;\nN-Carbamoylsarcosine;\nD-Asparagine'
    ],

    'alpha-Ketoisovaleric acid': [
        "Aminoadipic acid;\nAcetylhomoserine;\n(�)-2,2'-Iminobispropanoic acid",
    ],

    # -------------------------
    # PURINE BASES
    # -------------------------
    'Adenine': [
        'Guanine;\n2-Hydroxyadenine;\n8-Hydroxyadenine'
    ],

    # -------------------------
    # NEUROTRANSMITTER PRECURSORS
    # -------------------------
    'L-Dopa': [
        '4,5-seco-dopa'
    ],

    # -------------------------
    # CARBOHYDRATES
    # -------------------------
    'D-Fructose': [
        'Glucosamine;\nFructosamine;\nbeta-D-Glucosamine'
    ],

    # -------------------------
    # INOSITOL METABOLISM
    # -------------------------
    'Myoinositol': [
        'D-erythro-D-galacto-octitol',
        'Ribitol;\nD-Arabitol;\nL-Arabitol'
    ],

    # -------------------------
    # SULFUR METABOLISM
    # -------------------------
    'L-Cystine': [
        'L-Cystine',
        'Cysteineglutathione disulfide'
    ],

    # -------------------------
    # OTHER METABOLIC SIGNALS
    # -------------------------
    'Choline': [
        'sphingosylphosphorylcholine'
    ],

    'Vitamin C': [
        'cis-Aconitic acid;\ntrans-Aconitic acid;\nDehydroascorbic acid',
    ],

    'Bile acid': [
        '3alpha,7alpha-Dihydroxycoprostanic acid;\n3a,7a,12a-Trihydroxy-5b-cholestan-26-al;\n3alpha,7alpha,24(S)-trihydroxy-5beta-cholestan-27-al',
    ],

    # -------------------------
    # NOT OBSERVABLE IN MSI
    # -------------------------
    'Calcium': [],
    'Sodium': [],
    'Magnesium': [],
    'Zinc': [],
    'Copper': [],
    'Iron': [],
    'Cobalt': [],
    'Nickel': [],
    'Barium': [],
    'Cadmium': [],
    'Water': [],
    'Chloride': [],
    'Hydrogen carbonate': ['Hydrogen carbonate'],
}


from_metabolite_to_group = {
    'Butyrate': 'SCFAs',
    'Acetate': 'SCFAs',
    'Propionate': 'SCFAs',
    'Malate': 'TCA cycle',
    'Citrate': 'TCA cycle',
    'Fumarate': 'TCA cycle',
    'Succinate': 'TCA cycle',
}


def normalize_string(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def build_exchange_mapping(msi_list, targets):

    mapping = defaultdict(list)

    for target in targets:
        target_norm = normalize_string(target)

        for msi in msi_list:
            msi_lower = msi.lower()

            # --- Rule 1: direct match ---
            if target.lower() in msi_lower:
                mapping[target].append(msi)
                continue

            # --- Rule 2: keyword-based ---
            keywords = target.lower().split()

            if any(k in msi_lower for k in keywords):
                mapping[target].append(msi)
                continue

            # --- Rule 3: class-based ---
            if target.lower() in ["fatty acid", "palmitoleic acid", "myristic acid"]:
                if "fa(" in msi_lower:
                    mapping[target].append(msi)

            if "carnitine" in target.lower():
                if "carnitine" in msi_lower:
                    mapping[target].append(msi)

            if target.lower() in ["atp", "adp", "amp"]:
                if any(x in msi_lower for x in ["adenosine", "guanosine", "uridine", "hypoxanthine"]):
                    mapping[target].append(msi)

            if target.lower() in ["tca", "citrate", "succinate", "fumarate", "malate"]:
                if any(x in msi_lower for x in ["citrate", "malate", "fumarate", "succinate", "aconitic"]):
                    mapping[target].append(msi)

    return mapping
