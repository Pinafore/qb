# pylint: skip-file
# Source code below is from https://github.com/clips/pattern
# Copyright (c) 2011-2013 University of Antwerp, Belgium
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#   * Neither the name of Pattern nor the names of its
#     contributors may be used to endorse or promote products
#     derived from this software without specific prior written
#     permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import re

VERB, NOUN, ADJECTIVE, ADVERB = "VB", "NN", "JJ", "RB"
plural_prepositions = set((
    "about"  , "before" , "during", "of"   , "till" ,
    "above"  , "behind" , "except", "off"  , "to"   ,
    "across" , "below"  , "for"   , "on"   , "under",
    "after"  , "beneath", "from"  , "onto" , "until",
    "among"  , "beside" , "in"    , "out"  , "unto" ,
    "around" , "besides", "into"  , "over" , "upon" ,
    "at"     , "between", "near"  , "since", "with" ,
    "athwart", "betwixt",
               "beyond",
               "but",
               "by"))

# Inflection rules that are either:
# - general,
# - apply to a certain category of words,
# - apply to a certain category of words only in classical mode,
# - apply only in classical mode.
# Each rule is a (suffix, inflection, category, classic)-tuple.
plural_rules = [
       # 0) Indefinite articles and demonstratives.
    ((   r"^a$|^an$", "some"       , None, False),
     (     r"^this$", "these"      , None, False),
     (     r"^that$", "those"      , None, False),
     (      r"^any$", "all"        , None, False)
    ), # 1) Possessive adjectives.
    ((       r"^my$", "our"        , None, False),
     (     r"^your$", "your"       , None, False),
     (      r"^thy$", "your"       , None, False),
     (r"^her$|^his$", "their"      , None, False),
     (      r"^its$", "their"      , None, False),
     (    r"^their$", "their"      , None, False)
    ), # 2) Possessive pronouns.
    ((     r"^mine$", "ours"       , None, False),
     (    r"^yours$", "yours"      , None, False),
     (    r"^thine$", "yours"      , None, False),
     (r"^her$|^his$", "theirs"     , None, False),
     (      r"^its$", "theirs"     , None, False),
     (    r"^their$", "theirs"     , None, False)
    ), # 3) Personal pronouns.
    ((        r"^I$", "we"         , None, False),
     (       r"^me$", "us"         , None, False),
     (   r"^myself$", "ourselves"  , None, False),
     (      r"^you$", "you"        , None, False),
     (r"^thou$|^thee$", "ye"       , None, False),
     ( r"^yourself$", "yourself"   , None, False),
     (  r"^thyself$", "yourself"   , None, False),
     ( r"^she$|^he$", "they"       , None, False),
     (r"^it$|^they$", "they"       , None, False),
     (r"^her$|^him$", "them"       , None, False),
     (r"^it$|^them$", "them"       , None, False),
     (  r"^herself$", "themselves" , None, False),
     (  r"^himself$", "themselves" , None, False),
     (   r"^itself$", "themselves" , None, False),
     ( r"^themself$", "themselves" , None, False),
     (  r"^oneself$", "oneselves"  , None, False)
    ), # 4) Words that do not inflect.
    ((          r"$", ""  , "uninflected", False),
     (          r"$", ""  , "uncountable", False),
     (         r"s$", "s" , "s-singular" , False),
     (      r"fish$", "fish"       , None, False),
     (r"([- ])bass$", "\\1bass"    , None, False),
     (       r"ois$", "ois"        , None, False),
     (     r"sheep$", "sheep"      , None, False),
     (      r"deer$", "deer"       , None, False),
     (       r"pox$", "pox"        , None, False),
     (r"([A-Z].*)ese$", "\\1ese"   , None, False),
     (      r"itis$", "itis"       , None, False),
     (r"(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$", "\\1ose", None, False)
    ), # 5) Irregular plural forms (e.g., mongoose, oxen).
    ((     r"atlas$", "atlantes"   , None, True ),
     (     r"atlas$", "atlases"    , None, False),
     (      r"beef$", "beeves"     , None, True ),
     (   r"brother$", "brethren"   , None, True ),
     (     r"child$", "children"   , None, False),
     (    r"corpus$", "corpora"    , None, True ),
     (    r"corpus$", "corpuses"   , None, False),
     (      r"^cow$", "kine"       , None, True ),
     ( r"ephemeris$", "ephemerides", None, False),
     (  r"ganglion$", "ganglia"    , None, True ),
     (     r"genie$", "genii"      , None, True ),
     (     r"genus$", "genera"     , None, False),
     (  r"graffito$", "graffiti"   , None, False),
     (      r"loaf$", "loaves"     , None, False),
     (     r"money$", "monies"     , None, True ),
     (  r"mongoose$", "mongooses"  , None, False),
     (    r"mythos$", "mythoi"     , None, False),
     (   r"octopus$", "octopodes"  , None, True ),
     (      r"opus$", "opera"      , None, True ),
     (      r"opus$", "opuses"     , None, False),
     (       r"^ox$", "oxen"       , None, False),
     (     r"penis$", "penes"      , None, True ),
     (     r"penis$", "penises"    , None, False),
     ( r"soliloquy$", "soliloquies", None, False),
     (    r"testis$", "testes"     , None, False),
     (    r"trilby$", "trilbys"    , None, False),
     (      r"turf$", "turves"     , None, True ),
     (     r"numen$", "numena"     , None, False),
     (   r"occiput$", "occipita"   , None, True )
    ), # 6) Irregular inflections for common suffixes (e.g., synopses, mice, men).
    ((       r"man$", "men"        , None, False),
     (    r"person$", "people"     , None, False),
     (r"([lm])ouse$", "\\1ice"     , None, False),
     (     r"tooth$", "teeth"      , None, False),
     (     r"goose$", "geese"      , None, False),
     (      r"foot$", "feet"       , None, False),
     (      r"zoon$", "zoa"        , None, False),
     ( r"([csx])is$", "\\1es"      , None, False)
    ), # 7) Fully assimilated classical inflections
       #    (e.g., vertebrae, codices).
    ((        r"ex$", "ices" , "ex-ices" , False),
     (        r"ex$", "ices" , "ex-ices*", True ), # * = classical mode
     (        r"um$", "a"    ,    "um-a" , False),
     (        r"um$", "a"    ,    "um-a*", True ),
     (        r"on$", "a"    ,    "on-a" , False),
     (         r"a$", "ae"   ,    "a-ae" , False),
     (         r"a$", "ae"   ,    "a-ae*", True )
    ), # 8) Classical variants of modern inflections
       #    (e.g., stigmata, soprani).
    ((      r"trix$", "trices"     , None, True),
     (       r"eau$", "eaux"       , None, True),
     (       r"ieu$", "ieu"        , None, True),
     ( r"([iay])nx$", "\\1nges"    , None, True),
     (        r"en$", "ina"  ,  "en-ina*", True),
     (         r"a$", "ata"  ,   "a-ata*", True),
     (        r"is$", "ides" , "is-ides*", True),
     (        r"us$", "i"    ,    "us-i*", True),
     (        r"us$", "us "  ,   "us-us*", True),
     (         r"o$", "i"    ,     "o-i*", True),
     (          r"$", "i"    ,      "-i*", True),
     (          r"$", "im"   ,     "-im*", True)
    ), # 9) -ch, -sh and -ss take -es in the plural
       #    (e.g., churches, classes).
    ((   r"([cs])h$", "\\1hes"     , None, False),
     (        r"ss$", "sses"       , None, False),
     (         r"x$", "xes"        , None, False)
    ), # 10) -f or -fe sometimes take -ves in the plural
       #     (e.g, lives, wolves).
    (( r"([aeo]l)f$", "\\1ves"     , None, False),
     ( r"([^d]ea)f$", "\\1ves"     , None, False),
     (       r"arf$", "arves"      , None, False),
     (r"([nlw]i)fe$", "\\1ves"     , None, False),
    ), # 11) -y takes -ys if preceded by a vowel, -ies otherwise
       #     (e.g., storeys, Marys, stories).
    ((r"([aeiou])y$", "\\1ys"      , None, False),
     (r"([A-Z].*)y$", "\\1ys"      , None, False),
     (         r"y$", "ies"        , None, False)
    ), # 12) -o sometimes takes -os, -oes otherwise.
       #     -o is preceded by a vowel takes -os
       #     (e.g., lassos, potatoes, bamboos).
    ((         r"o$", "os",        "o-os", False),
     (r"([aeiou])o$", "\\1os"      , None, False),
     (         r"o$", "oes"        , None, False)
    ), # 13) Miltary stuff
       #     (e.g., Major Generals).
    ((         r"l$", "ls", "general-generals", False),
    ), # 14) Assume that the plural takes -s
       #     (cats, programmes, ...).
    ((          r"$", "s"          , None, False),)
]

# For performance, compile the regular expressions once:
plural_rules = [[(re.compile(r[0]), r[1], r[2], r[3]) for r in grp] for grp in plural_rules]

# Suffix categories.
plural_categories = {
    "uninflected": [
        "bison"      , "debris"     , "headquarters" , "news"       , "swine"        ,
        "bream"      , "diabetes"   , "herpes"       , "pincers"    , "trout"        ,
        "breeches"   , "djinn"      , "high-jinks"   , "pliers"     , "tuna"         ,
        "britches"   , "eland"      , "homework"     , "proceedings", "whiting"      ,
        "carp"       , "elk"        , "innings"      , "rabies"     , "wildebeest"
        "chassis"    , "flounder"   , "jackanapes"   , "salmon"     ,
        "clippers"   , "gallows"    , "mackerel"     , "scissors"   ,
        "cod"        , "graffiti"   , "measles"      , "series"     ,
        "contretemps",                "mews"         , "shears"     ,
        "corps"      ,                "mumps"        , "species"
        ],
    "uncountable": [
        "advice"     , "fruit"      , "ketchup"      , "meat"       , "sand"         ,
        "bread"      , "furniture"  , "knowledge"    , "mustard"    , "software"     ,
        "butter"     , "garbage"    , "love"         , "news"       , "understanding",
        "cheese"     , "gravel"     , "luggage"      , "progress"   , "water"
        "electricity", "happiness"  , "mathematics"  , "research"   ,
        "equipment"  , "information", "mayonnaise"   , "rice"
        ],
    "s-singular": [
        "acropolis"  , "caddis"     , "dais"         , "glottis"    , "pathos"       ,
        "aegis"      , "cannabis"   , "digitalis"    , "ibis"       , "pelvis"       ,
        "alias"      , "canvas"     , "epidermis"    , "lens"       , "polis"        ,
        "asbestos"   , "chaos"      , "ethos"        , "mantis"     , "rhinoceros"   ,
        "bathos"     , "cosmos"     , "gas"          , "marquis"    , "sassafras"    ,
        "bias"       ,                "glottis"      , "metropolis" , "trellis"
        ],
    "ex-ices": [
        "codex"      , "murex"      , "silex"
        ],
    "ex-ices*": [
        "apex"       , "index"      , "pontifex"     , "vertex"     ,
        "cortex"     , "latex"      , "simplex"      , "vortex"
        ],
    "um-a": [
        "agendum"    , "candelabrum", "desideratum"  , "extremum"   , "stratum"      ,
        "bacterium"  , "datum"      , "erratum"      , "ovum"
        ],
    "um-a*": [
        "aquarium"   , "emporium"   , "maximum"      , "optimum"    , "stadium"      ,
        "compendium" , "enconium"   , "medium"       , "phylum"     , "trapezium"    ,
        "consortium" , "gymnasium"  , "memorandum"   , "quantum"    , "ultimatum"    ,
        "cranium"    , "honorarium" , "millenium"    , "rostrum"    , "vacuum"       ,
        "curriculum" , "interregnum", "minimum"      , "spectrum"   , "velum"        ,
        "dictum"     , "lustrum"    , "momentum"     , "speculum"
        ],
    "on-a": [
        "aphelion"   , "hyperbaton" , "perihelion"   ,
        "asyndeton"  , "noumenon"   , "phenomenon"   ,
        "criterion"  , "organon"    , "prolegomenon"
        ],
    "a-ae": [
        "alga"       , "alumna"     , "vertebra"
        ],
    "a-ae*": [
        "abscissa"   , "aurora"     , "hyperbola"    , "nebula"     ,
        "amoeba"     , "formula"    , "lacuna"       , "nova"       ,
        "antenna"    , "hydra"      , "medusa"       , "parabola"
        ],
    "en-ina*": [
        "foramen"    , "lumen"      , "stamen"
    ],
    "a-ata*": [
        "anathema"   , "dogma"      , "gumma"        , "miasma"     , "stigma"       ,
        "bema"       , "drama"      , "lemma"        , "schema"     , "stoma"        ,
        "carcinoma"  , "edema"      , "lymphoma"     , "oedema"     , "trauma"       ,
        "charisma"   , "enema"      , "magma"        , "sarcoma"    ,
        "diploma"    , "enigma"     , "melisma"      , "soma"       ,
        ],
    "is-ides*": [
        "clitoris"   , "iris"
        ],
    "us-i*": [
        "focus"      , "nimbus"     , "succubus"     ,
        "fungus"     , "nucleolus"  , "torus"        ,
        "genius"     , "radius"     , "umbilicus"    ,
        "incubus"    , "stylus"     , "uterus"
        ],
    "us-us*": [
        "apparatus"  , "hiatus"     , "plexus"       , "status"
        "cantus"     , "impetus"    , "prospectus"   ,
        "coitus"     , "nexus"      , "sinus"        ,
        ],
    "o-i*": [
        "alto"       , "canto"      , "crescendo"    , "soprano"    ,
        "basso"      , "contralto"  , "solo"         , "tempo"
        ],
    "-i*": [
        "afreet"     , "afrit"      , "efreet"
        ],
    "-im*": [
        "cherub"     , "goy"        , "seraph"
        ],
    "o-os": [
        "albino"     , "dynamo"     , "guano"        , "lumbago"    , "photo"        ,
        "archipelago", "embryo"     , "inferno"      , "magneto"    , "pro"          ,
        "armadillo"  , "fiasco"     , "jumbo"        , "manifesto"  , "quarto"       ,
        "commando"   , "generalissimo",                "medico"     , "rhino"        ,
        "ditto"      , "ghetto"     , "lingo"        , "octavo"     , "stylo"
        ],
    "general-generals": [
        "Adjutant"   , "Brigadier"  , "Lieutenant"   , "Major"      , "Quartermaster",
        "adjutant"   , "brigadier"  , "lieutenant"   , "major"      , "quartermaster"
        ]
}


def pluralize(word, pos=NOUN, custom={}, classical=True):
    """ Returns the plural of a given word, e.g., child => children.
        Handles nouns and adjectives, using classical inflection by default
        (i.e., where "matrix" pluralizes to "matrices" and not "matrixes").
        The custom dictionary is for user-defined replacements.
    """
    if word in custom:
        return custom[word]
    # Recurse genitives.
    # Remove the apostrophe and any trailing -s,
    # form the plural of the resultant noun, and then append an apostrophe (dog's => dogs').
    if word.endswith(("'", "'s")):
        w = word.rstrip("'s")
        w = pluralize(w, pos, custom, classical)
        if w.endswith("s"):
            return w + "'"
        else:
            return w + "'s"
    # Recurse compound words
    # (e.g., Postmasters General, mothers-in-law, Roman deities).
    w = word.replace("-", " ").split(" ")
    if len(w) > 1:
        if w[1] == "general" or \
           w[1] == "General" and \
           w[0] not in plural_categories["general-generals"]:
            return word.replace(w[0], pluralize(w[0], pos, custom, classical))
        elif w[1] in plural_prepositions:
            return word.replace(w[0], pluralize(w[0], pos, custom, classical))
        else:
            return word.replace(w[-1], pluralize(w[-1], pos, custom, classical))
    # Only a very few number of adjectives inflect.
    n = range(len(plural_rules))
    if pos.startswith(ADJECTIVE):
        n = [0, 1]
    # Apply pluralization rules.
    for i in n:
        for suffix, inflection, category, classic in plural_rules[i]:
            # A general rule, or a classic rule in classical mode.
            if category is None:
                if not classic or (classic and classical):
                    if suffix.search(word) is not None:
                        return suffix.sub(inflection, word)
            # A rule pertaining to a specific category of words.
            if category is not None:
                if word in plural_categories[category] and (not classic or (classic and classical)):
                    if suffix.search(word) is not None:
                        return suffix.sub(inflection, word)
    return word