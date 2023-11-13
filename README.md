
<h1 align="center">TASP-CNN Current State</h1>

## First Paper

Three classes

  - [Madrid](https://github.com/luperezsal/TASP-CNN/commit/525a28e028b495d9c0932dd692c88ad806df4de4):  [DGX] Testing models 

## Second Paper
Two classes, filtering areas.
  - [Madrid](#github-readme-profile-category)
  - [UK](https://github.com/luperezsal/TASP-CNN/commit/f8c3ab6a410e80339d937ee4055c36a7a78a3e4f):  [FEATURE] Last updates 

## Third 
Add Sin/Cos to Hour and categories
  - [Madrid](https://github.com/luperezsal/TASP-CNN/commit/a2980a170222f85a5b36334742f536ace4ff59db): [FEATURE] Add knn missing model
  - UK:
    - [All models](https://github.com/luperezsal/TASP-CNN/commit/07d2b0d3f5ffbcbcd6dbae4c76b585de9b16c621): [FEATURE] Retrain all models with 6x5 matrix 
    - [Cornwall](https://github.com/luperezsal/TASP-CNN/commit/848e5e907cfa3f1e5090088acbffbb3ec162a98a): [FEATURE] Add Cornwall 6x5
  - [Victoria](https://github.com/luperezsal/TASP-CNN/commit/2a94da53c18dbe707cd8719c13cc4b9065e2ad23): [FEATURE] Add diagram Victoria vs Adelaida 


### Removing features
  - Cornwall:
    - [Environmental](https://github.com/luperezsal/TASP-CNN/commit/43d81accfe691b92f2db2b18aa70dfb50ba7dc61)
    - [Environmental-Vehicle](https://github.com/luperezsal/TASP-CNN/commit/fb56ce81b07257d6514a355f59e1d74e756289af)


To view F1-Score by model for a single city (report summary test file) run this scrpit:

    import pandas as pd
    
    report_summary_path = '2023-10-13-13 21 02.csv'
    report_summary = pd.read_csv(report_summary_path)
    report_summary[(report_summary['accident_type'] == 'Slight') | (report_summary['accident_type'] == 'Assistance')].sort_values(['model', 'accident_type'])

