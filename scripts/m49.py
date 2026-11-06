"""UN M49 Level-2 sub-regions (JMP 'master regions', JMP Methodology 2017 Table 10). Hand-coded from the M49 standard."""
M49 = {
 'Eastern Africa': 'BDI COM DJI ERI ETH KEN MDG MWI MUS MYT MOZ REU RWA SYC SOM SSD UGA TZA ZMB ZWE',
 'Middle Africa': 'AGO CMR CAF TCD COG COD GNQ GAB STP',
 'Southern Africa': 'BWA SWZ LSO NAM ZAF',
 'Western Africa': 'BEN BFA CPV CIV GMB GHA GIN GNB LBR MLI MRT NER NGA SHN SEN SLE TGO',
 'Northern Africa': 'DZA EGY LBY MAR SDN TUN ESH',
 'Western Asia': 'ARM AZE BHR CYP GEO IRQ ISR JOR KWT LBN OMN QAT SAU PSE SYR TUR ARE YEM',
 'Central Asia': 'KAZ KGZ TJK TKM UZB',
 'Southern Asia': 'AFG BGD BTN IND IRN MDV NPL PAK LKA',
 'Eastern Asia': 'CHN HKG MAC PRK JPN MNG KOR',
 'South-eastern Asia': 'BRN KHM IDN LAO MYS MMR PHL SGP THA TLS VNM',
 'Caribbean': 'AIA ATG ABW BHS BRB BES VGB CYM CUB CUW DMA DOM GRD GLP HTI JAM MTQ MSR PRI BLM KNA LCA MAF VCT SXM TTO TCA VIR',
 'Central America': 'BLZ CRI SLV GTM HND MEX NIC PAN',
 'South America': 'ARG BOL BRA CHL COL ECU FLK GUF GUY PRY PER SUR URY VEN',
 'Northern America': 'BMU CAN GRL SPM USA',
 'Northern Europe': 'ALA DNK EST FRO FIN GGY ISL IRL IMN JEY LVA LTU NOR SJM SWE GBR',
 'Eastern Europe': 'BLR BGR CZE HUN POL MDA ROU RUS SVK UKR',
 'Southern Europe': 'ALB AND BIH HRV GIB GRC VAT ITA MLT MNE MKD PRT SMR SRB SVN ESP XKX',
 'Western Europe': 'AUT BEL FRA DEU LIE LUX MCO NLD CHE',
 'Australia and New Zealand': 'AUS NZL NFK',
 'Melanesia': 'FJI NCL PNG SLB VUT',
 'Micronesia': 'GUM KIR MHL FSM NRU MNP PLW',
 'Polynesia': 'ASM COK PYF NIU PCN WSM TKL TON TUV WLF',
}
ISO2M49 = {iso: reg for reg, s in M49.items() for iso in s.split()}
M49_L1 = {'Africa': ['Eastern Africa', 'Middle Africa', 'Southern Africa', 'Western Africa', 'Northern Africa'],
          'Americas': ['Caribbean', 'Central America', 'South America', 'Northern America'],
          'Asia': ['Western Asia', 'Central Asia', 'Southern Asia', 'Eastern Asia', 'South-eastern Asia'],
          'Europe': ['Northern Europe', 'Eastern Europe', 'Southern Europe', 'Western Europe'],
          'Oceania': ['Australia and New Zealand', 'Melanesia', 'Micronesia', 'Polynesia']}
SUB2L1 = {s: r for r, subs in M49_L1.items() for s in subs}
