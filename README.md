
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
  - [Madrid](hhttps://github.com/luperezsal/TASP-CNN/commit/5850dd2f931f861cc9c385c534bcf05365bbabee): [FIX] Fix Madrid normal v2
  - UK:
    - [All models](https://github.com/luperezsal/TASP-CNN/commit/07d2b0d3f5ffbcbcd6dbae4c76b585de9b16c621): [FEATURE] Retrain all models with 6x5 matrix 
    - [Cornwall](https://github.com/luperezsal/TASP-CNN/commit/848e5e907cfa3f1e5090088acbffbb3ec162a98a): [FEATURE] Add Cornwall 6x5
  - [Victoria](https://github.com/luperezsal/TASP-CNN/commit/2a94da53c18dbe707cd8719c13cc4b9065e2ad23): [FEATURE] Add diagram Victoria vs Adelaida 


### Removing categories
  - Cornwall:
    - [Environmental](https://github.com/luperezsal/TASP-CNN/commit/43d81accfe691b92f2db2b18aa70dfb50ba7dc61)
    - [Vehicle Best](https://github.com/luperezsal/TASP-CNN/commit/82d4a5a79005f30faf94371e4508a690d7a96621)
    - [Environmental-Vehicle](https://github.com/luperezsal/TASP-CNN/commit/fb56ce81b07257d6514a355f59e1d74e756289af)

### Removing features

  - Less importance:
    - UK - Cornwall:
      - [Age Of Vehicle](https://github.com/luperezsal/TASP-CNN/commit/9c14e7fa13e5c7948e1aa35773747e95e96bf099)
    - Australia - Victoria:
      - [Age Of Vehicle](https://github.com/luperezsal/TASP-CNN/commit/2119acfede6b1295dabf69c617d2b661c7f06c4b)
    - Spain - Madrid:
      - [Tipo Persona](https://github.com/luperezsal/TASP-CNN/commit/efb5d9b0f65b360f2da924ea298826f679edc62a)
  - More importance:
    - UK - Cornwall:
      - [Type Of Vehicle](https://github.com/luperezsal/TASP-CNN/commit/48cda10ad64e76d13ff48bd30ca71820124a156d)
    - Australia - Victoria:
      - [First Point Of Impact](https://github.com/luperezsal/TASP-CNN/commit/1dbf36d70e1ce6136f6af4aca94385070616bc33)
    - Spain - Madrid:
      - [Tipo Vehículo](https://github.com/luperezsal/TASP-CNN/commit/7561ec5e68cc38e539a44d9db033e49b698ce43d)
  - Combinations (Less | More):
    - UK - Cornwall:
      - [Age Of Vehicle | Type Of Vehicle](https://github.com/luperezsal/TASP-CNN/commit/788a7514415fe297afa43d8124db79622efff0be)
    - Australia - Victoria:
      - [Age Of Vehicle | First Point Of Impact](https://github.com/luperezsal/TASP-CNN/commit/95e5556662a832c6755e10812f108e2b6cfb0fb6)
    - Spain - Madrid:
      - [Tipo Persona | Tipo Vehículo](https://github.com/luperezsal/TASP-CNN/commit/7561ec5e68cc38e539a44d9db033e49b698ce43d)

To view F1-Score by model for a single city (report summary test file) run this scrpit:

    import pandas as pd
    
    report_summary_path = '2023-10-13-13 21 02.csv'
    report_summary = pd.read_csv(report_summary_path)
    report_summary[(report_summary['accident_type'] == 'Slight') | (report_summary['accident_type'] == 'Assistance')].sort_values(['model', 'accident_type'])

