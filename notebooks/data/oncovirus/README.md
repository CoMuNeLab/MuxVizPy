# OncoVirus tutorial data

These four edge lists form one complete `N` sample from the data accompanying:

F. Zambelli, V. Pancaldi and M. De Domenico, “Unraveling the Network Signatures
of Oncogenicity in Virus–Human Protein–Protein Interactions”, *Entropy* 27 (2025),
1248. [doi:10.3390/e27121248](https://doi.org/10.3390/e27121248).

The sample is line 33 of `data/MultilayerIndexes/n.txt` in the
[OncoVirus repository](https://github.com/francescozambelli/OncoVirus), containing
virus indices 18, 77, 16 and 43. The repository classifies all four viruses as
non-oncogenic. Each CSV is copied without row changes from
`data/processed/SyntheticViruses/original/<virus>/edges.csv`.

The rows are undirected interactions between human protein symbols. The study built each
layer by retaining the human proteins targeted by a virus, adding their first neighbours
in the human protein interaction network, and keeping every interaction between those
proteins. The original repository describes its material as MIT licensed. The article's
data statement refers readers to that repository and to the BIOstring dataset described
by Ghavasieh et al., “Multiscale statistical physics of the pan-viral interactome
unravels the systemic nature of SARS-CoV-2 infections”, *Communications Physics* 4
(2021), 83.

Source directories:

- `gallid-herpesvirus-2.csv` comes from
  `Gallid_herpesvirus_2__strain_Chicken-Md5-ATCC_VR-987_`.
- `yaba-monkey-tumour-virus.csv` comes from
  `Yaba_monkey_tumor_virus__strain_VR587_`.
- `equine-herpesvirus-2.csv` comes from
  `Equine_herpesvirus_2__strain_86-87_`.
- `human-parechovirus-2.csv` comes from
  `Human_parechovirus_2__strain_Williamson_`.
