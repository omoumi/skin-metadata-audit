# skin-metadata-audit

Shared toolkit behind a set of per-dataset audits of skin-lesion image datasets that come with
patient metadata. Every dataset goes through the same checks, so that the results can be put side
by side; how far they are comparable is limited by differences in labels, fields and collection
between the datasets.

No image model is trained. The audit answers five questions per dataset:

1. **How complete is the metadata?** Missing rate per field, per class and per acquisition group;
   missingness patterns; size of the complete-case subsets.
2. **Is missingness informative?** Association of each field's missing indicator with the label
   (bias-corrected Cramer's V, Bergsma 2013, with group-bootstrap intervals), and how well the indicators alone
   predict the label.
3. **Which non-image shortcuts may exist?** Acquisition source, image type and image resolution as
   predictors of the label.
4. **How much signal do the metadata values carry?** Grouped-CV balanced accuracy of the metadata
   alone, against a permuted-label null. Scores are averaged over repeated CV runs with an
   approximate 95% interval (corrected variance of Nadeau and Bengio 2003, as used for repeated CV
   by Bouckaert and Frank 2004). When lesions sit inside patients, p-values come from a null that
   swaps whole patients (labels of one patient are alike); they are Holm-adjusted (Holm 1979) over the
   feature sets. One model class (gradient boosting with fixed settings) is used, so a result without a
   significant gain does not show that no such signal exists.
5. **What explains the missingness, and are complete rows representative?** For each field, models
   predict that it is missing from the other fields with and without the label. Two tests of the
   label's contribution are run (a conditional randomization test, Candès et al. 2018, and a swap
   of whole CV groups, e.g. patients, within strata), and a field is reported as label-dependent only when both reject after Holm
   adjustment; a field that is not reported may still depend on the label, since the power of this
   rule is limited. Standardized mean differences (Yang and Dalton 2012; |SMD| > 0.1 as a
   conventional mark of imbalance, Austin 2009) compare the complete-case subsets with the whole
   dataset. A limit, documented in `mechanism.py`: when the label summarizes something in the other
   fields that the model cannot learn, these tests cannot tell that apart from real label
   dependence (Shah and Peters 2020); the reported label R2 gives a rough indication of how much this may matter.

Results are associations, not causes: an acquisition source that predicts the label may reflect
a real difference between the patient populations of two centres. The false-positive rate, power
and interval coverage of the tests were checked by simulation on synthetic data only
([validation/README.md](validation/README.md)). Each `summary.json` records the package versions
(scikit-learn pinned to 1.6.1) and the md5 of the input files.

## Layout

```
skinaudit/       the toolkit: adapters (one per dataset), statistics, shortcut and missingness tests,
                 duplicate detection, figures, per-dataset README and cross-dataset comparison
notebooks/       the Kaggle notebooks that produced the published results
duplicates/      image pairs confirmed as duplicates, one file per dataset, with the reason for each pair
validation/      simulation scripts and their results
tests/           unit tests on small synthetic data
```

## Shared taxonomy (comparison only)

Each audit keeps its native labels. For the comparison, every native label is mapped once to
level 1 (malignant / benign / indeterminate) and level 2 (MEL, NV, BCC, SCC, AK, BKL, DF, VASC,
OTHER, UNSPECIFIED); mixed native classes (e.g. AKIEC) are flagged as ambiguous. Conventions such
as "actinic keratosis is indeterminate" follow the ISIC-Dx hierarchy and are applied the same way
everywhere (`skinaudit/taxonomy.py`).

## References

- Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. Stat Med. 2009;28(25):3083-3107. doi:10.1002/sim.3697
- Bergsma W. A bias-correction for Cramér's V and Tschuprow's T. J Korean Stat Soc. 2013;42(3):323-328. doi:10.1016/j.jkss.2012.10.002
- Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. In: Advances in Knowledge Discovery and Data Mining (PAKDD 2004). Lecture Notes in Computer Science, vol 3056. Springer; 2004:3-12. doi:10.1007/978-3-540-24775-3_3
- Candès E, Fan Y, Janson L, Lv J. Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection. J R Stat Soc Series B Stat Methodol. 2018;80(3):551-577. doi:10.1111/rssb.12265
- Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70.
- Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52(3):239-281. doi:10.1023/A:1024068626366
- Phipson B, Smyth GK. Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. Stat Appl Genet Mol Biol. 2010;9(1):Article 39. doi:10.2202/1544-6115.1585
- Shah RD, Peters J. The hardness of conditional independence testing and the generalised covariance measure. Ann Stat. 2020;48(3):1514-1538. doi:10.1214/19-AOS1857
- Tomczak M, Tomczak E. The need to report effect size estimates revisited. An overview of some recommended measures of effect size. Trends Sport Sci. 2014;1(21):19-25.
- Yang D, Dalton JE. A unified approach to measuring the effect size between two groups using SAS. SAS Global Forum 2012; paper 335-2012.

## License

Code: GPL-3.0-only, Copyright (C) 2026 Mohammad Hossein Omoumi ([LICENSE](LICENSE), [NOTICE](NOTICE)).
The datasets are not included and keep their own licenses; the tables, figures and text in each
per-dataset repository carry a license that follows the dataset's terms (see its `LICENSE-CONTENT.md`).

## Citing

Citation metadata is in [CITATION.cff](CITATION.cff).
