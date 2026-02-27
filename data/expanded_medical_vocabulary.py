"""
Expanded medieval medical Latin vocabulary.

Sources: Dioscorides (De Materia Medica), Galen (Constantine the African),
Circa Instans (Platearius), Macer Floridus (De Viribus Herbarum),
Antidotarium Nicolai.

Each category maps lemma -> list of inflected surface forms.
The skeleton matcher strips vowels and maps consonants to classes, so
only forms that differ in consonant structure need separate entries.
Duplicate skeletons are harmless (map to same dict entry), but missing
a skeleton means a potential match is lost.

Deduplication: all lemmas checked against existing vocabulary in Phase 4
(LATIN_PLANT_NAMES), Phase 5 (EXPANDED_PLANT_NAMES, EXPANDED_BODY_WORDS,
CONDITION_WORDS, PREPARATION_WORDS, EXPANDED_SUBSTANCE_WORDS,
EXPANDED_PROPERTY_WORDS), and Phase 6 (ALL_NEW_VOCAB: GALENIC_TERMS,
ASTROLOGICAL_TERMS, ANIMAL_INGREDIENTS, MINERAL_INGREDIENTS,
DIAGNOSTIC_TERMS, EXTRA_VERBS, EXTRA_NOUNS, EXTRA_ADJECTIVES).

Phase 12.5  ·  Voynich Convergence Attack
"""

# ============================================================================
# PLANT NAMES (~50 new lemmas, ~150 surface forms)
# ============================================================================
# Excluded: all plants already in Phase 4 LATIN_PLANT_NAMES (~30) and
# Phase 5 EXPANDED_PLANT_NAMES (~70). Only genuinely new entries below.

MEDICAL_PLANT_NAMES = {
    # Shrubs and trees
    'althaea': ['althaea', 'althaeae', 'althaeam', 'althaearum'],
    'amygdalum': ['amygdalum', 'amygdali', 'amygdalo', 'amygdalorum'],
    'balsamum': ['balsamum', 'balsami', 'balsamo', 'balsamorum'],
    'brassica': ['brassica', 'brassicae', 'brassicam', 'brassicarum'],
    'colchicum': ['colchicum', 'colchici', 'colchico'],
    'euphorbia': ['euphorbia', 'euphorbiae', 'euphorbiam', 'euphorbiarum'],
    'gladiolus': ['gladiolus', 'gladioli', 'gladiolo', 'gladiolorum'],
    'glycyrrhiza': ['glycyrrhiza', 'glycyrrhizae', 'glycyrrhizam'],
    'linum': ['linum', 'lini', 'lino', 'linorum'],
    'olibanum': ['olibanum', 'olibani', 'olibano'],
    'pastinaca': ['pastinaca', 'pastinacae', 'pastinacam', 'pastinacarum'],
    'piper': ['piper', 'piperis', 'pipere', 'piperibus'],
    'primula': ['primula', 'primulae', 'primulam'],
    'prunus': ['prunus', 'pruni', 'pruno', 'prunorum'],
    'saponaria': ['saponaria', 'saponariae', 'saponariam'],
    'sinapis': ['sinapis', 'sinapis', 'sinapem', 'sinapibus'],
    # Herbs and flowers
    'acorus': ['acorus', 'acori', 'acoro'],
    'asparagus': ['asparagus', 'asparagi', 'asparago', 'asparagorum'],
    'berberis': ['berberis', 'berberis', 'berberem', 'berberibus'],
    'buglossa': ['buglossa', 'buglossae', 'buglossam'],
    'camphora': ['camphora', 'camphorae', 'camphorem', 'camphoram'],
    'capparis': ['capparis', 'capparis', 'capparem', 'capparibus'],
    'citrullus': ['citrullus', 'citrulli', 'citrullo'],
    'cyperus': ['cyperus', 'cyperi', 'cypero'],
    'diptamnus': ['diptamnus', 'diptamni', 'diptamno'],
    'epithymum': ['epithymum', 'epithymi', 'epithymo'],
    'eruca': ['eruca', 'erucae', 'erucam'],
    'foeniculum': ['foeniculum', 'foeniculi', 'foeniculo'],
    'isatis': ['isatis', 'isatidis', 'isatidem', 'isatide'],
    'nardus': ['nardus', 'nardi', 'nardo'],
    'psyllium': ['psyllium', 'psyllii', 'psyllio'],
    'rhaponticum': ['rhaponticum', 'rhapontici', 'rhapontico'],
    'rubia': ['rubia', 'rubiae', 'rubiam'],
    'ruscus': ['ruscus', 'rusci', 'rusco'],
    'saxifraga': ['saxifraga', 'saxifragae', 'saxifragam'],
    'scolopendrium': ['scolopendrium', 'scolopendrii', 'scolopendrio'],
    'senna': ['senna', 'sennae', 'sennam'],
    'spica': ['spica', 'spicae', 'spicam', 'spicarum'],
    'squilla': ['squilla', 'squillae', 'squillam'],
    'tamarindus': ['tamarindus', 'tamarindi', 'tamarindo'],
    'terebinthina': ['terebinthina', 'terebinthinae', 'terebinthinam'],
    'theriaca': ['theriaca', 'theriacae', 'theriacam'],
    'turbith': ['turbith', 'turbithi', 'turbitho'],
    'zedoaria': ['zedoaria', 'zedoariae', 'zedoariam'],
}

# ============================================================================
# ANATOMICAL TERMS (~55 new lemmas, ~200 surface forms)
# ============================================================================
# Excluded: terms already in Phase 4 LATIN_BODY_WORDS (~18),
# Phase 5 EXPANDED_BODY_WORDS (+25), and Phase 6 EXTRA_NOUNS (~50).

MEDICAL_ANATOMICAL_TERMS = {
    # Head and face
    'abdomen': ['abdomen', 'abdominis', 'abdomine', 'abdominibus'],
    'calcaneum': ['calcaneum', 'calcanei', 'calcaneo'],
    'calvaria': ['calvaria', 'calvariae', 'calvariam'],
    'clavicula': ['clavicula', 'claviculae', 'claviculam', 'clavicularum'],
    'costa': ['costa', 'costae', 'costam', 'costarum', 'costis'],
    'cranium': ['cranium', 'cranii', 'cranio'],
    'cubitus': ['cubitus', 'cubiti', 'cubito', 'cubitorum'],
    'femur': ['femur', 'femoris', 'femore', 'femoribus'],
    'fibula': ['fibula', 'fibulae', 'fibulam', 'fibularum'],
    'gingiva': ['gingiva', 'gingivae', 'gingivam', 'gingivarum'],
    'guttur': ['guttur', 'gutturis', 'gutture', 'gutturibus'],
    'humerus': ['humerus', 'humeri', 'humero', 'humerorum'],
    'ileum': ['ileum', 'ilei', 'ileo'],
    'intestinum': ['intestinum', 'intestini', 'intestino', 'intestinorum'],
    'iugulum': ['iugulum', 'iuguli', 'iugulo'],
    'lumbus': ['lumbus', 'lumbi', 'lumbo', 'lumborum', 'lumbis'],
    'mamma': ['mamma', 'mammae', 'mammam', 'mammarum'],
    'mandibula': ['mandibula', 'mandibulae', 'mandibulam'],
    'matrix': ['matrix', 'matricis', 'matricem', 'matrice'],
    'membrum': ['membrum', 'membri', 'membro', 'membrorum', 'membris'],
    'naris': ['naris', 'naris', 'narem', 'naribus'],
    'nodus': ['nodus', 'nodi', 'nodo', 'nodorum', 'nodis'],
    'occiput': ['occiput', 'occipitis', 'occipite'],
    'palatum': ['palatum', 'palati', 'palato'],
    'patella': ['patella', 'patellae', 'patellam'],
    'pelvis': ['pelvis', 'pelvis', 'pelvem', 'pelve'],
    'pleura': ['pleura', 'pleurae', 'pleuram'],
    'pollex': ['pollex', 'pollicis', 'pollicem', 'pollice'],
    'pulmo': ['pulmo', 'pulmonis', 'pulmonem', 'pulmone', 'pulmonibus'],
    'pupilla': ['pupilla', 'pupillae', 'pupillam'],
    'radius': ['radius', 'radii', 'radio'],
    'scapula': ['scapula', 'scapulae', 'scapulam', 'scapularum'],
    'sinus': ['sinus', 'sinus', 'sinui', 'sinuum'],
    'spina': ['spina', 'spinae', 'spinam', 'spinarum'],
    'sternum': ['sternum', 'sterni', 'sterno'],
    'stomachus': ['stomachus', 'stomachi', 'stomacho'],
    'talus': ['talus', 'tali', 'talo'],
    'tempora': ['tempora', 'temporum', 'temporibus'],
    'testiculus': ['testiculus', 'testiculi', 'testiculo', 'testiculorum'],
    'thorax': ['thorax', 'thoracis', 'thoracem', 'thorace'],
    'tibia': ['tibia', 'tibiae', 'tibiam'],
    'trachea': ['trachea', 'tracheae', 'tracheam'],
    'tympanum': ['tympanum', 'tympani', 'tympano'],
    'umbilicus': ['umbilicus', 'umbilici', 'umbilico'],
    'unguis': ['unguis', 'unguis', 'unguem', 'ungue', 'unguibus'],
    'uvula': ['uvula', 'uvulae', 'uvulam'],
    'ventriculus': ['ventriculus', 'ventriculi', 'ventriculo', 'ventriculorum'],
    'vertebra': ['vertebra', 'vertebrae', 'vertebram', 'vertebrarum', 'vertebris'],
    'viscera': ['viscera', 'viscerum', 'visceribus'],
    'vulva': ['vulva', 'vulvae', 'vulvam'],
}

# ============================================================================
# PHARMACEUTICAL PREPARATION TERMS (~35 new lemmas, ~100 surface forms)
# ============================================================================
# Excluded: terms already in Phase 5 PREPARATION_WORDS and EXPANDED_SUBSTANCE_WORDS.

MEDICAL_PHARMACEUTICAL_TERMS = {
    'antidotum': ['antidotum', 'antidoti', 'antidoto', 'antidotorum'],
    'apozema': ['apozema', 'apozematis', 'apozemate'],
    'cataplasma': ['cataplasma', 'cataplasmatis', 'cataplasmate'],
    'ceratum': ['ceratum', 'cerati', 'cerato', 'ceratorum'],
    'clystere': ['clystere', 'clysteris', 'clystere', 'clysteribus'],
    'collyrium': ['collyrium', 'collyrii', 'collyrio', 'collyriorum'],
    'confectio': ['confectio', 'confectionis', 'confectionem', 'confectione'],
    'decoctum': ['decoctum', 'decocti', 'decocto', 'decoctorum'],
    'electuarium': ['electuarium', 'electuarii', 'electuario'],
    'elixir': ['elixir', 'elixiris', 'elixire'],
    'epithema': ['epithema', 'epithematis', 'epithemate'],
    'errhinum': ['errhinum', 'errhini', 'errhino'],
    'extractum': ['extractum', 'extracti', 'extracto'],
    'fomentum': ['fomentum', 'fomenti', 'fomento', 'fomentorum'],
    'fumigatio': ['fumigatio', 'fumigationis', 'fumigationem', 'fumigatione'],
    'gargarisma': ['gargarisma', 'gargarismatis', 'gargarismate'],
    'infusum': ['infusum', 'infusi', 'infuso'],
    'inunctio': ['inunctio', 'inunctionis', 'inunctionem', 'inunctione'],
    'lavacrum': ['lavacrum', 'lavacri', 'lavacro'],
    'linimentum': ['linimentum', 'linimenti', 'linimento', 'linimentorum'],
    'lotio': ['lotio', 'lotionis', 'lotionem', 'lotione'],
    'masticatorium': ['masticatorium', 'masticatorii', 'masticatorio'],
    'mixtura': ['mixtura', 'mixturae', 'mixturam', 'mixturarum'],
    'oxymel': ['oxymel', 'oxymelis', 'oxymele'],
    'pessarium': ['pessarium', 'pessarii', 'pessario'],
    'sirupus': ['sirupus', 'sirupi', 'sirupo', 'siruporum'],
    'species': ['species', 'speciei', 'specierum', 'speciebus'],
    'suppositorium': ['suppositorium', 'suppositorii', 'suppositorio'],
    'tinctura': ['tinctura', 'tincturae', 'tincturam'],
    'trochiscus': ['trochiscus', 'trochisci', 'trochisco', 'trochiscorum'],
    'vehiculum': ['vehiculum', 'vehiculi', 'vehiculo'],
}

# ============================================================================
# DISEASE / SYMPTOM TERMS (~45 new lemmas, ~150 surface forms)
# ============================================================================
# Excluded: conditions already in Phase 5 CONDITION_WORDS (~35) and
# Phase 6 DIAGNOSTIC_TERMS (~35).

MEDICAL_DISEASE_TERMS = {
    'abscessus': ['abscessus', 'abscessus', 'abscessum', 'abscessui'],
    'aegritudo': ['aegritudo', 'aegritudinis', 'aegritudinem', 'aegritudine'],
    'arthritis': ['arthritis', 'arthritidis', 'arthritidem', 'arthritide'],
    'cancer': ['cancer', 'cancri', 'cancro', 'cancrorum'],
    'cataractus': ['cataractus', 'cataracti', 'cataracto'],
    'cephalalgia': ['cephalalgia', 'cephalalgiae', 'cephalalgiam'],
    'cholera': ['cholera', 'cholerae', 'choleram'],
    'colica': ['colica', 'colicae', 'colicam'],
    'diabetes': ['diabetes', 'diabetis', 'diabetem', 'diabete'],
    'diarrhoea': ['diarrhoea', 'diarrhoeae', 'diarrhoeam'],
    'dropsia': ['dropsia', 'dropsiae', 'dropsiam'],
    'erysipelas': ['erysipelas', 'erysipelatis', 'erysipelate'],
    'fistula': ['fistula', 'fistulae', 'fistulam', 'fistularum'],
    'fluxus': ['fluxus', 'fluxus', 'fluxum', 'fluxui'],
    'fractura': ['fractura', 'fracturae', 'fracturam', 'fracturarum'],
    'gangraena': ['gangraena', 'gangraenae', 'gangraenam'],
    'gonorrhoea': ['gonorrhoea', 'gonorrhoeae', 'gonorrhoeam'],
    'gutta': ['gutta', 'guttae', 'guttam', 'guttarum'],
    'haemorrhagia': ['haemorrhagia', 'haemorrhagiae', 'haemorrhagiam'],
    'hernia': ['hernia', 'herniae', 'herniam', 'herniarum'],
    'hydrops': ['hydrops', 'hydropis', 'hydropem', 'hydrope'],
    'icterus': ['icterus', 'icteri', 'ictero'],
    'inflammatio': ['inflammatio', 'inflammationis', 'inflammationem', 'inflammatione'],
    'lepra': ['lepra', 'leprae', 'lepram'],
    'lethargia': ['lethargia', 'lethargiae', 'lethargiam'],
    'lithiasis': ['lithiasis', 'lithiasis', 'lithiasem', 'lithiase'],
    'lumbago': ['lumbago', 'lumbaginis', 'lumbaginem', 'lumbagiine'],
    'mania': ['mania', 'maniae', 'maniam'],
    'morbus': ['morbus', 'morbi', 'morbum', 'morbo', 'morborum', 'morbis'],
    'obstipatio': ['obstipatio', 'obstipationis', 'obstipationem', 'obstipatione'],
    'ophthalmia': ['ophthalmia', 'ophthalmiae', 'ophthalmiam'],
    'paralysis': ['paralysis', 'paralysis', 'paralysem', 'paralyse'],
    'pestilentia': ['pestilentia', 'pestilentiae', 'pestilentiam'],
    'phlegmon': ['phlegmon', 'phlegmonis', 'phlegmonem', 'phlegmone'],
    'pleuritis': ['pleuritis', 'pleuritidis', 'pleuritidem', 'pleuritide'],
    'podagra': ['podagra', 'podagrae', 'podagram'],
    'scabies': ['scabies', 'scabiei', 'scabiem'],
    'spasmus': ['spasmus', 'spasmi', 'spasmo', 'spasmorum'],
    'struma': ['struma', 'strumae', 'strumam', 'strumarum'],
    'suffocatio': ['suffocatio', 'suffocationis', 'suffocationem', 'suffocatione'],
    'tumor': ['tumor', 'tumoris', 'tumorem', 'tumore', 'tumoribus'],
    'tussis': ['tussis', 'tussis', 'tussem', 'tusse'],
    'ulcus': ['ulcus', 'ulceris', 'ulcere', 'ulceribus'],
    'variola': ['variola', 'variolae', 'variolam', 'variolarum'],
    'vulnus': ['vulnus', 'vulneris', 'vulnerem', 'vulnere', 'vulneribus'],
}

# ============================================================================
# DOSAGE / MEASUREMENT TERMS (~12 new lemmas, ~40 surface forms)
# ============================================================================
# Excluded: terms already in Phase 5 DOSAGE_WORDS.

MEDICAL_DOSAGE_TERMS = {
    'cochlear': ['cochlear', 'cochlearis', 'cochleare', 'cochlearibus'],
    'cyathus': ['cyathus', 'cyathi', 'cyatho', 'cyathorum'],
    'drachma': ['drachma', 'drachmae', 'drachmam', 'drachmarum'],
    'manipulus': ['manipulus', 'manipuli', 'manipulo', 'manipulorum'],
    'mensura': ['mensura', 'mensurae', 'mensuram', 'mensurarum'],
    'modicum': ['modicum', 'modici', 'modico'],
    'obulus': ['obulus', 'obuli', 'obulo', 'obulorum'],
    'pugillus': ['pugillus', 'pugilli', 'pugillo'],
    'quantum': ['quantum', 'quanti', 'quanto'],
    'scrupulus': ['scrupulus', 'scrupuli', 'scrupulo', 'scrupulorum'],
    'semuncia': ['semuncia', 'semunciae', 'semunciam'],
    'uncia': ['uncia', 'unciae', 'unciam', 'unciarum'],
}

# ============================================================================
# PROCESS VERBS (~40 new lemmas, ~240 surface forms)
# ============================================================================
# Excluded: verbs already in Phase 6 EXTRA_VERBS (~31 forms).
# Each verb includes: infinitive, 3sg present, 3pl present, imperative,
# past participle (m/f), gerund.

MEDICAL_PROCESS_VERBS = {
    'addere': ['addere', 'addit', 'addunt', 'adde', 'additum', 'addendo'],
    'admiscere': ['admiscere', 'admiscet', 'admiscent', 'admisce',
                  'admixtum', 'admiscendo'],
    'clarificare': ['clarificare', 'clarificat', 'clarificant', 'clarifica',
                    'clarificatum', 'clarificando'],
    'coagulare': ['coagulare', 'coagulat', 'coagulant', 'coagula',
                  'coagulatum', 'coagulando'],
    'colligere': ['colligere', 'colligit', 'colligunt', 'collige',
                  'collectum', 'colligendo'],
    'comminuere': ['comminuere', 'comminuit', 'comminuunt', 'comminue',
                   'comminutum', 'comminuendo'],
    'componere': ['componere', 'componit', 'componunt', 'compone',
                  'compositum', 'componendo'],
    'conficere': ['conficere', 'conficit', 'conficiunt', 'confice',
                  'confectum', 'conficiendo'],
    'conterere': ['conterere', 'conterit', 'conterunt', 'contere',
                  'contritum', 'conterendo'],
    'coquere': ['coquere', 'coquit', 'coquunt', 'coque',
                'coctum', 'coquendo'],
    'decoquere': ['decoquere', 'decoquit', 'decoquunt', 'decoque',
                  'decoctum', 'decoquendo'],
    'defaecare': ['defaecare', 'defaecat', 'defaecant', 'defaeca',
                  'defaecatum', 'defaecando'],
    'destillare': ['destillare', 'destillat', 'destillant', 'destilla',
                   'destillatum', 'destillando'],
    'emplastrare': ['emplastrare', 'emplastrat', 'emplastrant', 'emplastra',
                    'emplastratum', 'emplastrando'],
    'evacuare': ['evacuare', 'evacuat', 'evacuant', 'evacua',
                 'evacuatum', 'evacuando'],
    'excoriare': ['excoriare', 'excoriat', 'excoriant', 'excoria',
                  'excoriatum', 'excoriando'],
    'exprimere': ['exprimere', 'exprimit', 'exprimunt', 'exprime',
                  'expressum', 'exprimendo'],
    'filtrare': ['filtrare', 'filtrat', 'filtrant', 'filtra',
                 'filtratum', 'filtrando'],
    'humectare': ['humectare', 'humectat', 'humectant', 'humecta',
                  'humectatum', 'humectando'],
    'immergere': ['immergere', 'immergit', 'immergunt', 'immerge',
                  'immersum', 'immergendo'],
    'incidere': ['incidere', 'incidit', 'incidunt', 'incide',
                 'incisum', 'incidendo'],
    'incorporare': ['incorporare', 'incorporat', 'incorporant', 'incorpora',
                    'incorporatum', 'incorporando'],
    'infundere': ['infundere', 'infundit', 'infundunt', 'infunde',
                  'infusum', 'infundendo'],
    'inspissare': ['inspissare', 'inspissat', 'inspissant', 'inspissa',
                   'inspissatum', 'inspissando'],
    'lavare': ['lavare', 'lavat', 'lavant', 'lava',
               'lavatum', 'lavando'],
    'lenire': ['lenire', 'lenit', 'leniunt', 'leni',
               'lenitum', 'leniendo'],
    'liquefare': ['liquefare', 'liquefacit', 'liquefaciunt', 'liquefac',
                  'liquefactum', 'liquefaciendo'],
    'macerare': ['macerare', 'macerat', 'macerant', 'macera',
                 'maceratum', 'macerando'],
    'mollificare': ['mollificare', 'mollificat', 'mollificant', 'mollifica',
                    'mollificatum', 'mollificando'],
    'mundificare': ['mundificare', 'mundificat', 'mundificant', 'mundifica',
                    'mundificatum', 'mundificando'],
    'permiscere': ['permiscere', 'permiscet', 'permiscent', 'permisce',
                   'permixtum', 'permiscendo'],
    'praeparare': ['praeparare', 'praeparat', 'praeparant', 'praepara',
                   'praeparatum', 'praeparando'],
    'refrigerare': ['refrigerare', 'refrigerat', 'refrigerant', 'refrigera',
                    'refrigeratum', 'refrigerando'],
    'restringere': ['restringere', 'restringit', 'restringunt', 'restringe',
                    'restrictum', 'restringendo'],
    'scarificare': ['scarificare', 'scarificat', 'scarificant', 'scarifica',
                    'scarificatum', 'scarificando'],
    'stillare': ['stillare', 'stillat', 'stillant', 'stilla',
                 'stillatum', 'stillando'],
    'sublimare': ['sublimare', 'sublimat', 'sublimant', 'sublima',
                  'sublimatum', 'sublimando'],
    'suffumigare': ['suffumigare', 'suffumigat', 'suffumigant', 'suffumiga',
                    'suffumigatum', 'suffumigando'],
    'terere': ['terere', 'terit', 'terunt', 'tere',
               'tritum', 'terendo'],
    'torrere': ['torrere', 'torret', 'torrent', 'torre',
                'tostum', 'torrendo'],
    'triturare': ['triturare', 'triturat', 'triturant', 'tritura',
                  'trituratum', 'triturando'],
    'ungere': ['ungere', 'ungit', 'ungunt', 'unge',
               'unctum', 'ungendo'],
    'urere': ['urere', 'urit', 'urunt', 'ure',
              'ustum', 'urendo'],
}

# ============================================================================
# ALL CATEGORIES (for iteration by corpus builder)
# ============================================================================

ALL_MEDICAL_CATEGORIES = {
    'plant_names': MEDICAL_PLANT_NAMES,
    'anatomical_terms': MEDICAL_ANATOMICAL_TERMS,
    'pharmaceutical_terms': MEDICAL_PHARMACEUTICAL_TERMS,
    'disease_terms': MEDICAL_DISEASE_TERMS,
    'dosage_terms': MEDICAL_DOSAGE_TERMS,
    'process_verbs': MEDICAL_PROCESS_VERBS,
}

# Frequency weights per category (how many times each form appears in corpus).
# Kept low (1-2) to avoid displacing template-generated content that builds
# the transition matrix. The main goal is to get each form into the skeleton
# index; transition probabilities come from appearing in template contexts.
CATEGORY_WEIGHTS = {
    'plant_names': 2,         # Recipe subjects
    'anatomical_terms': 1,    # "apply to [body part]" phrases
    'pharmaceutical_terms': 2, # Preparation instructions
    'disease_terms': 1,       # Named in indications
    'dosage_terms': 1,        # Quantity phrases
    'process_verbs': 2,       # Recipe instructions
}
