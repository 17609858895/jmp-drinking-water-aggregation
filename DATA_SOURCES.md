# Input data

The analyses use WHO/UNICEF Joint Monitoring Programme (JMP) **household world files**, not the separate country-profile workbooks. Download the files yourself from the official JMP service. Raw workbooks are excluded from this repository. This code release does not change the data provider's terms.

The [JMP downloads index](https://washdata.org/data/downloads) links to the [Household World file](https://washdata.org/data/country/WLD/household/download). That dynamic link may later point to a newer edition. For the study inputs, use the edition-specific links below and verify SHA256 before preparing data.

| Edition | Official download | Local filename in `data/raw/` | Bytes | SHA256 |
|---|---|---|---:|---|
| 2025 | [JMP file 4671](https://washdata.org/media/4671/download) | `JMP_2025_WLD.xlsx` | 25098506 | `5b26a1fb97c5a8f91916ddb1a59664072455cb44223e88eec536a841c5bbeed7` |
| 2023 | [JMP file 3716](https://washdata.org/media/3716/download) | `JMP_2023_WLD.xlsx` | 18531969 | `b19fe9fa3fbc51f513835f15ec2009eb129d0d4b8c4ed2081c344054adb79943` |
| 2021 | [JMP file 2178](https://washdata.org/media/2178/download) | `JMP_2021_WLD.xlsx` | 20478718 | `ffca1c3b08b6f30fd774793759f0c4f553f8b489c81f5f855f8b5d370f6d8d05` |

These hashes were checked against the study's local workbooks during code release. The web downloads index was accessible at release; the binary links were not re-downloaded as part of the smoke checks. A provider revision or unavailable archived link therefore requires checking the original edition with JMP, rather than accepting a changed file silently.

```text
data/raw/
  JMP_2025_WLD.xlsx
  JMP_2023_WLD.xlsx  # only for retrospective edition analyses
  JMP_2021_WLD.xlsx  # only for retrospective edition analyses
```

Run `python run.py check-data --editions 2025 2023 2021` after download. The main 2025 analysis reads the unformatted `wat`, `san` and `hyg` sheets; preparation also preserves the `mh` sheet. Population values in these files are in thousands. Derived coverage is expressed in percent; population-equivalent aggregation differences are reported in millions of people and are accounting quantities, not household classifications.

`15_vintage_load.py` harmonises older sheet column names, including the 2021 `_n` national suffix. Income and development classifications missing from earlier editions are inherited from the 2025 file by country. Water, sanitation and hygiene outcomes remain edition-specific. The edition comparisons are retrospective at matched reference years.

No external covariate dataset or trained-model download is required. The small `scripts/m49.py` lookup supplies UN M49 regional groupings for the imputation hierarchy; country codes are matched to the JMP `iso3` field. See the [UN M49 overview](https://unstats.un.org/unsd/methodology/m49/) for the regional classification. Country/territory reporting units are retained from JMP.
