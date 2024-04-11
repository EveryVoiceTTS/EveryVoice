# /*
# *  Arpabet-to-IPA - converting Arpabet to IPA.
# *
# * @author		Waldeilson Eder dos Santos. Adapted by Aidan Pine.
# * @copyright 	Copyright (c) 2015 Waldeilson Eder dos Santos
# * @copyright 	Copyright (c) 2024 National Research Council Canada
# * @link			https://github.com/wwesantos/arpabet-to-ipa
# * @package     	arpabet-to-ipa
# *
# * The MIT License (MIT)
# *
# * Permission is hereby granted, free of charge, to any person obtaining a copy
# * of this software and associated documentation files (the "Software"), to deal
# * in the Software without restriction, including without limitation the rights
# * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# * copies of the Software, and to permit persons to whom the Software is
# * furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in all
# * copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# * SOFTWARE.
# */

""" A Simple Arpabet to IPA conversion
    adapted by Aidan Pine from https://github.com/wwesantos/arpabet-to-ipa/blob/master/src/App.php

    Except changed so that 1 adds a pre-vocalic primary stress symbol and 2 adds a secondary stress symbol.
    These flout the IPA standard of placing stress symbols before syllables, but this is more computationally
    tractable (in the absence of reliable syllabification algorithms for all datasets that include Arpabet).

"""
from g2p.mappings import Mapping, Rule
from g2p.transducer import Transducer

ARPABET_LOOKUP = {
    "AO": "ɔ",
    "AO0": "ɔ",
    "AO1": "ˈɔ",
    "AO2": "ˌɔ",
    "AA": "ɑ",
    "AA0": "ɑ",
    "AA1": "ˈɑ",
    "AA2": "ˌɑ",
    "IY": "i",
    "IY0": "i",
    "IY1": "ˈi",
    "IY2": "ˌi",
    "UW": "u",
    "UW0": "u",
    "UW1": "ˈu",
    "UW2": "ˌu",
    "EH": "e",  # modern versions use 'e' instead of 'ɛ'
    "EH0": "e",  # ɛ
    "EH1": "ˈe",  # ɛ
    "EH2": "ˌe",  # ɛ
    "IH": "ɪ",
    "IH0": "ɪ",
    "IH1": "ˈɪ",
    "IH2": "ˌɪ",
    "UH": "ʊ",
    "UH0": "ʊ",
    "UH1": "ˈʊ",
    "UH2": "ˌʊ",
    "AH": "ʌ",
    "AH0": "ə",
    "AH1": "ˈʌ",
    "AH2": "ˌʌ",
    "AE": "æ",
    "AE0": "æ",
    "AE1": "ˈæ",
    "AE2": "ˌæ",
    "AX": "ə",
    "AX0": "ə",
    "AX1": "ˈə",
    "AX2": "ˌə",
    # /*
    # Vowels - Diphthongs
    # Arpabet	IPA	Word Examples
    # EY		eɪ	say (S EY1); eight (EY1 T)
    # AY		aɪ	my (M AY1); why (W AY1); ride (R AY1 D)
    # OW		oʊ	show (SH OW1); coat (K OW1 T)
    # AW		aʊ	how (HH AW1); now (N AW1)
    # OY		ɔɪ	boy (B OY1); toy (T OY1)
    # */
    "EY": "eɪ",
    "EY0": "eɪ",
    "EY1": "ˈeɪ",
    "EY2": "ˌeɪ",
    "AY": "aɪ",
    "AY0": "aɪ",
    "AY1": "ˈaɪ",
    "AY2": "ˌaɪ",
    "OW": "oʊ",
    "OW0": "oʊ",
    "OW1": "ˈoʊ",
    "OW2": "ˌoʊ",
    "AW": "aʊ",
    "AW0": "aʊ",
    "AW1": "ˈaʊ",
    "AW2": "ˌaʊ",
    "OY": "ɔɪ",
    "OY0": "ɔɪ",
    "OY1": "ˈɔɪ",
    "OY2": "ˌɔɪ",
    # /*
    # Consonants - Stops
    # Arpabet	IPA	Word Examples
    # P		p	pay (P EY1)
    # B		b	buy (B AY1)
    # T		t	take (T EY1 K)
    # D		d	day (D EY1)
    # K		k	key (K IY1)
    # G		ɡ	go (G OW1)
    # */
    "P": "p",
    "B": "b",
    "T": "t",
    "D": "d",
    "K": "k",
    "G": "g",
    # /*
    # Consonants - Affricates
    # Arpabet	IPA	Word Examples
    # CH		tʃ	chair (CH EH1 R)
    # JH		dʒ	just (JH AH1 S T); gym (JH IH1 M)
    # */
    # 	'CH' : 'tʃ',
    # 	'JH' : 'dʒ',
    # /*
    # Consonants - Fricatives
    # Arpabet	IPA	Word Examples
    # F		f	for (F AO1 R)
    # V		v	very (V EH1 R IY0)
    # TH		θ	thanks (TH AE1 NG K S); Thursday (TH ER1 Z D EY2)
    # DH		ð	that (DH AE1 T); the (DH AH0); them (DH EH1 M)
    # S		s	say (S EY1)
    # Z		z	zoo (Z UW1)
    # SH		ʃ	show (SH OW1)
    # ZH		ʒ	measure (M EH1 ZH ER0); pleasure (P L EH1 ZH ER)
    # HH		h	house (HH AW1 S)
    # */
    "F": "f",
    "V": "v",
    "TH": "θ",
    "DH": "ð",
    "S": "s",
    "Z": "z",
    "SH": "ʃ",
    "ZH": "ʒ",
    "HH": "h",
    # /*
    # Consonants - Nasals
    # Arpabet	IPA	Word Examples
    # M		m	man (M AE1 N)
    # N		n	no (N OW1)
    # NG		ŋ	sing (S IH1 NG)
    # */
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    # /*
    #  Consonants - Liquids
    # Arpabet	IPA		Word Examples
    # L		ɫ OR l	late (L EY1 T)
    # R		r OR ɹ	run (R AH1 N)
    # */
    "L": "l",
    "R": "r",
    # /*
    #  Vowels - R-colored vowels
    # Arpabet			IPA	Word Examples
    # ER				ɝ	her (HH ER0); bird (B ER1 D); hurt (HH ER1 T), nurse (N ER1 S)
    # AXR				ɚ	father (F AA1 DH ER); coward (K AW1 ER D)
    # The following R-colored vowels are contemplated above
    # EH R			ɛr	air (EH1 R); where (W EH1 R); hair (HH EH1 R)
    # UH R			ʊr	cure (K Y UH1 R); bureau (B Y UH1 R OW0), detour (D IH0 T UH1 R)
    # AO R			ɔr	more (M AO1 R); bored (B AO1 R D); chord (K AO1 R D)
    # AA R			ɑr	large (L AA1 R JH); hard (HH AA1 R D)
    # IH R or IY R	ɪr	ear (IY1 R); near (N IH1 R)
    # AW R			aʊr	This seems to be a rarely used r-controlled vowel. In some dialects flower (F L AW1 R; in other dialects F L AW1 ER0)
    # */
    "ER": "ɜr",
    "ER0": "ɜr",
    "ER1": "ˈɜr",
    "ER2": "ˌɜr",
    "AXR": "ər",
    "AXR0": "ər",
    "AXR1": "ˈər",
    "AXR2": "ˌər",
    # /*
    # Consonants - Semivowels
    # Arpabet	IPA	Word Examples
    # Y		j	yes (Y EH1 S)
    # W		w	way (W EY1)
    # */
    "W": "w",
    "Y": "j",
}

mapping = Mapping(
    rules=[Rule(rule_input=k, rule_output=v) for k, v in ARPABET_LOOKUP.items()],
    in_lang="arpabet",
    out_lang="ipa",
    rule_ordering="apply-longest-first",
    out_delimiter="",
)
ARPABET_TO_IPA_TRANSDUCER = Transducer(mapping)
