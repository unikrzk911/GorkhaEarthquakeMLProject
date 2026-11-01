# Classifying Earthquake-Triggered Landslide Susceptibility

**Using machine learning to predict landslide-prone slopes in the 2015 Gorkha earthquake region, Nepal**

A Random Forest trained on terrain, ground shaking and geology identifies the slopes that failed in the 2015 Gorkha earthquake with a ROC-AUC of 0.890, scored only on districts it never saw during training. Models trained the conventional way, on landslides mapped before the earthquake, catch only 57–74% of the earthquake landslides, against 81–82% for models trained on the earthquake itself.

Everything comes from one notebook, `landslide_susceptibility.ipynb`. It compares Logistic Regression, an RBF Support Vector Machine and a Random Forest against a Frequency Ratio baseline at 90 m resolution, for the 25 April 2015 M7.8 main shock and the 12 May 2015 M7.3 aftershock.

---

## Main results

Leave-one-district-out cross-validation, 2,477 co-seismic landslides against 2,477 non-landslide points:

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Kappa |
|---|---|---|---|---|---|
| Frequency Ratio (baseline) | 0.855 | 0.843 | 0.763 | 0.780 | 0.527 |
| Logistic Regression | 0.864 | 0.843 | 0.789 | 0.793 | 0.577 |
| SVM (RBF) | 0.877 | 0.860 | 0.800 | 0.804 | 0.600 |
| **Random Forest** | **0.890** | **0.877** | **0.812** | **0.811** | **0.624** |

Random Forest wins overall, but SVM wins in six of the nine held-out districts. Removing the ground-shaking features lowers AUC by 0.025 (SVM) to 0.061 (Random Forest).

### Earthquake model (A) vs pre-earthquake model (B)

Scored on the same co-seismic points at a 0.5 threshold:

| Model | A detected | A false alarms | B detected | B false alarms |
|---|---|---|---|---|
| Frequency Ratio | 83.8% | 31.1% | 81.5% | 46.6% |
| Logistic Regression | 81.0% | 23.3% | 74.0% | 31.9% |
| SVM (RBF) | 82.2% | 22.2% | 58.8% | 21.3% |
| Random Forest | 80.6% | 18.2% | 57.2% | 27.5% |

Model A still beats Model B when shaking is removed from both, so missing shaking data explains only part of the gap.

### What drives the predictions

SHAP ranks PGA × sin(slope) first, then main-shock PGA, slope, relief and elevation. The model learned thresholds of about 0.6 g of shaking and 30° of slope, matching field observations.

### Map validation

On a map trained without Sindhupalchok:

| Class | Area | Unseen Sindhupalchok slides (986) | Bahrabise slides (551) |
|---|---|---|---|
| Low | 40% | 0.2% | 5.1% |
| Moderate | 30% | 6.9% | 17.4% |
| High | 20% | 24.5% | 21.8% |
| Very High | 10% | 68.4% | 55.7% |
| Prediction-rate AUC | | 0.907 | 0.836 |

One test fell short: inside the small Bahrabise area, the point-based AUC is only 0.59–0.67. The diagnostic (cell 6.8) shows the cause is mostly the inventory, not the model: on the same negatives, Random Forest scores 0.665 on Bahrabise slides but 0.838 on ICIMOD slides from the same area.

---

## Folder layout

```
Gorkha Earthquake Project/
├── landslide_susceptibility.ipynb   main notebook (steps 1–7)
├── requirements.txt                 Python packages
├── Datasets/                        input data (see below)
│   ├── Landslide data of 14 earthquake affected districts of Nepal/data/14dist_ls.shp
│   ├── Nepal's Gorkha Earthquake 2015 induced landslide data of Bahrabise, Sindhupalchok/data/Bahrabise_slide_megh.shp
│   ├── Geology of Nepal/data/Geology.shp
│   ├── srtm-cgiar-nepal-boundary-dem-90m/srtm_cgiar_nepal_boundary.img
│   ├── grid_mainshock.xml
│   └── grid_aftershock.xml
└── outputs/                         created by the notebook
    ├── figures/                     PNG figures
    ├── *.csv                        features, predictions, result tables
    └── susceptibility_*.tif         susceptibility maps (GeoTIFF, UTM 45N, 90 m)
```

Cell 1.4 checks that every input file is present before any processing starts.

---

## Data

| Dataset | Use | Source |
|---|---|---|
| Landslide inventory, 14 districts (4,747 polygons) | Labels | [ICIMOD](https://rds.icimod.org/metadata/5e19b3a6-910e-4537-a49e-bce2e5da2e78) |
| Bahrabise landslide inventory (551 polygons) | External test | [ICIMOD](https://rds.icimod.org/metadata/1b4f293a-c131-415f-8e39-a58d2e8bae23) |
| Geology of Nepal (856 polygons) | Rock type | [ICIMOD](https://rds.icimod.org/metadata/e0e362b7-0da2-46b3-9323-f301d3b281b5) |
| SRTM 90 m DEM | Terrain features | [CGIAR-CSI](https://srtm.csi.cgiar.org) |
| ShakeMap, main shock | Ground shaking (PGA) | [USGS us20002926](https://earthquake.usgs.gov/product/shakemap/us20002926/atlas/1594162031303/download/grid.xml) |
| ShakeMap, aftershock | Ground shaking (PGA) | [USGS us20002ejl](https://earthquake.usgs.gov/product/shakemap/us20002ejl/atlas/1594162075191/download/grid.xml) |

ICIMOD data are used under their download agreements; USGS ShakeMaps are public domain.

---

## How to run

1. Install Python 3 (tested with 3.14.8).
2. Create the environment in the project folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Open the notebook in PyCharm, VS Code or Jupyter with the `.venv` kernel.
4. **Run All** in one session. Don't restart the kernel part-way, as later steps reuse earlier results.

### Runtime

| Step | Content | Time |
|---|---|---|
| 1–4 | Setup, data cleaning, features, exploratory analysis | ~4 min |
| 5 | Model comparison with nested cross-validation | ~6.5 min |
| 6 | Extra experiments (ablation, 1:3 ratio, Model A vs B, Bahrabise) | ~25.5 min |
| 7 | SHAP, susceptibility map, validation | ~6 min |

A full run takes about 43 minutes.

---

## Method in brief

- **Labels:** landslides first seen on images from 25 April to 30 June 2015 are co-seismic (2,477); earlier ones are pre-earthquake (1,857).
- **Samples:** one point per landslide; non-landslide points are random, at least 150 m from any mapped landslide.
- **Features:** elevation, slope, aspect, curvature and relief from the DEM; main-shock and aftershock PGA; PGA × sin(slope); geology.
- **Validation:** nested leave-one-district-out cross-validation, so every score comes from unseen districts.
- **Map:** all 90 m cells are classed by area into Low (40%), Moderate (30%), High (20%) and Very High (10%).

---

## Reproducibility

Every random step uses `SEED = 42`, and a second run reproduced every number exactly. Tested with numpy 2.5.4, pandas 3.0.6, geopandas 1.2.0, rasterio 1.5.2, scikit-learn 1.9.1, matplotlib 3.11.2 and shap 0.53.0.

---

## Limitations

- The model learned ICIMOD's mapping style; an inventory mapped differently (Bahrabise) scores lower.
- Inventory dates are image dates, not failure dates.
- The median landslide is smaller than one 90 m cell.
- Tested on one earthquake only.
- The map is a research result, not an official hazard map. A Low class does not mean a slope is safe.
