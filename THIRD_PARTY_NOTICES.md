# Third-party source

This research handoff preserves upstream source and license notices.

| Component | Source | Version / provenance | License file |
|---|---|---|---|
| LEXA | https://anonymous.4open.science/r/LEXA-5913 | Downloaded source; core model/effects SHA match the fitted pilot metadata | `LEXA-5913/LICENSE` |
| CPA | https://github.com/facebookresearch/CPA | `50e283a7a1b3d43470d8adc5e11d3be4c10a534c` | `lexa_text_harness/benchmarks/pilot_matched_baselines/vendor/CPA-main/LICENSE` |
| CODEX | https://github.com/sschrod/CODEX | `0e5f8ff3b10d21b582cf91ea6cf5398fe8eeab99` | `lexa_text_harness/literature/tdc_leaderboard/CODEX-master/LICENSE` |

Upstream training helpers can use different data, stopping criteria and evaluation protocols. The local adapters and README document the actual protocol used here; these local scores are not reproductions of published leaderboard numbers.

GEO sciPlex2 inputs are public research data. Raw input files are not included in Git; the downloader, original accession, source URLs and checksums are retained. The portable replay artifact is a derived group-mean scoring export of the already-unblinded development comparison.
