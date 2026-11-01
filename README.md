# Classifying Earthquake-Triggered Landslide Susceptibility

**Using machine learning to predict landslide-prone slopes in the 2015 Gorkha earthquake region, Nepal**

Unique Rajak (17626107) · STW7072CEM Machine Learning, March Intake 2026 · Softwarica College of IT & E-Commerce (Coventry University)

A Random Forest trained on terrain, ground shaking and geology picks out the slopes that failed in the 2015 Gorkha earthquake with a ROC-AUC of 0.890, scored only on districts it never saw during training. A model built the conventional way, from landslides mapped before the earthquake, does markedly worse on the same slopes: its three machine-learning versions catch only 57 to 74% of the earthquake landslides, where the same three models trained on earthquake landslides catch 81 to 82%.

Everything here comes from one notebook, `landslide_susceptibility.ipynb`. It compares Logistic Regression, an RBF Support Vector Machine and a Random Forest against a Frequency Ratio baseline at 90 m resolution, for the 25 April 2015 M7.8 main shock and the 12 May 2015 M7.3 aftershock.

---

## Main results

Leave-one-district-out cross-validation, 2,477 co-seismic landslides against 2,477 non-landslide points:

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Kappa |
|---|---|---|---|---|---|
| Frequency Ratio (baseline) | 0.855 | 0.843 | 0.763 | 0.780 | 0.527 |
| Logistic Regression | 0.864 | 0.843 | 0.789 | 0.793 | 0.577 |
| SVM (RBF) | 0.877 | 0.860 | 0.800 | 0.804 | 0.600 |
| **Random Forest** | **0.890** | **0.877** | **0.812** | **0.811** | **0.624** |

The Random Forest wins on the pooled predictions, but not everywhere: the SVM has the higher AUC in six of the nine held-out district groups. Shaking carries real information on top of the terrain. Removing the three ground-shaking features costs between 0.025 (SVM) and 0.061 (Random Forest) of AUC.

### Earthquake model (A) against pre-earthquake model (B)

Both are scored on the same co-seismic points at a 0.5 threshold:

| Model | A detected | A false alarms | B detected | B false alarms |
|---|---|---|---|---|
| Frequency Ratio | 83.8% | 31.1% | 81.5% | 46.6% |
| Logistic Regression | 81.0% | 23.3% | 74.0% | 31.9% |
| SVM (RBF) | 82.2% | 22.2% | 58.8% | 21.3% |
| Random Forest | 80.6% | 18.2% | 57.2% | 27.5% |

Model B's AUC is 0.083 to 0.172 lower. Missing shaking data explains only part of the gap: with shaking removed from Model A as well, A still beats B by 0.028 (LR) to 0.111 (RF).

### What drives the predictions

SHAP ranks PGA × sin(slope) first, then main-shock PGA, slope, relief and elevation. The three shaking-based features hold 47% of the total mean |SHAP|. The model places its thresholds at about 0.55–0.6 g of shaking and 28–30° of slope, close to the 0.6 g and 30° found by field mapping.

### Map validation

On a map trained without Sindhupalchok:

| Class | Area | Unseen Sindhupalchok slides (986) | Bahrabise slides (551) |
|---|---|---|---|
| Low | 40% | 0.2% | 5.1% |
| Moderate | 30% | 6.9% | 17.4% |
| High | 20% | 24.5% | 21.8% |
| Very High | 10% | 68.4% | 55.7% |
| Prediction-rate AUC | | 0.907 | 0.836 |

The High and Very High classes cover 30% of the area but hold 92.9% of Sindhupalchok's unseen landslides and 77.5% of the independent Bahrabise inventory.

One test failed. Scored point by point against non-landslide locations inside the small Bahrabise area, the models reach an AUC of only 0.59 to 0.67, well short of the 0.80 target. The notebook's diagnostic (cell 6.8) traces most of that drop to the Bahrabise inventory itself rather than to the model. Against the same negatives, the Random Forest scores 0.665 on the Bahrabise slides but 0.838 on ICIMOD's own co-seismic slides from the same area, and LR and SVM move the same way. Bahrabise slides are smaller (median 2,887 m², ICIMOD 5,010 m²) and sit on gentler ground (median slope 32°, ICIMOD 37°).

---

## Folder layout

```
Gorkha Earthquake Project/
├── landslide_susceptibility.ipynb   main notebook (steps 1–7)
├── requirements.txt                 Python packages
├── README.md                        this file
├── Datasets/                        input data (see "Data" below)
│   ├── Landslide data of 14 earthquake affected districts of Nepal/data/14dist_ls.shp
│   ├── Nepal's Gorkha Earthquake 2015 induced landslide data of Bahrabise, Sindhupalchok/data/Bahrabise_slide_megh.shp
│   ├── Geology of Nepal/data/Geology.shp
│   ├── srtm-cgiar-nepal-boundary-dem-90m/srtm_cgiar_nepal_boundary.img
│   ├── grid_mainshock.xml
│   └── grid_aftershock.xml
└── outputs/                         created by the notebook
    ├── figures/                     all PNG figures
    ├── *.csv                        feature table, predictions, result tables
    ├── landslides_clean.gpkg        cleaned inventories
    └── susceptibility_*.tif         susceptibility maps (GeoTIFF, UTM 45N, 90 m)
```

The notebook expects the data at exactly these paths, relative to the project folder (cell 1.2). Cell 1.4 stops with a list of missing files if any are absent, so a wrong folder shows up before any processing starts.

---

## Data

| Dataset | Use | Source |
|---|---|---|
| Landslide inventory of 14 earthquake-affected districts (4,747 polygons; 13 districts actually present) | Labels: co-seismic and pre-earthquake landslides | ICIMOD Regional Database System, https://rds.icimod.org/metadata/5e19b3a6-910e-4537-a49e-bce2e5da2e78 |
| Bahrabise landslide inventory (551 polygons) | Independent external test | ICIMOD RDS, https://rds.icimod.org/metadata/1b4f293a-c131-415f-8e39-a58d2e8bae23 |
| Geology of Nepal (856 polygons) | Rock-type feature | ICIMOD RDS, https://rds.icimod.org/metadata/e0e362b7-0da2-46b3-9323-f301d3b281b5 |
| SRTM 90 m DEM (CGIAR-CSI), clipped to Nepal | Elevation, slope, aspect, curvature, relief | CGIAR-CSI SRTM, https://srtm.csi.cgiar.org |
| USGS ShakeMap, main shock us20002926 | Peak ground acceleration (PGA) | https://earthquake.usgs.gov/product/shakemap/us20002926/atlas/1594162031303/download/grid.xml |
| USGS ShakeMap, aftershock us20002ejl | PGA | https://earthquake.usgs.gov/product/shakemap/us20002ejl/atlas/1594162075191/download/grid.xml |

Each ICIMOD dataset ships with a `Data Download Agreement.pdf` in its folder, and that agreement sets the terms of use. USGS ShakeMaps are public domain.

---

## How to run

### Locally (tested on Windows 11 with PyCharm)

1. Install Python 3 (tested with 3.14.8).
2. Open a terminal in the project folder and create the environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
3. Open `landslide_susceptibility.ipynb` in PyCharm, VS Code or Jupyter (`jupyter notebook`), choose the `.venv` kernel, and check that the working directory is the project folder.
4. Run all cells from top to bottom (**Run All**). Don't restart the kernel part-way: steps 6 and 7 reuse the terrain grid and the models built earlier in the same session.

Cell 1.1 sets `plt.style.use("default")`, so saved figures have white backgrounds even when the IDE uses a dark theme.

### Runtime

Measured on the author's laptop:

| Step | Content | Time |
|---|---|---|
| 1 | Setup, paths, input check | < 10 s |
| 2 | Load and clean the inventories | ~1 min |
| 3 | Terrain derivatives, sampling, feature table | ~2 min |
| 4 | Exploratory analysis and PCA | < 1 min |
| 5 | Frequency Ratio, LR, SVM, RF with nested leave-one-district-out CV | ~6.5 min |
| 6 | Ablation (6.0 min), 1:3 ratio (14.7 min), Model A vs B (4.4 min), Bahrabise external test | ~25.5 min |
| 7 | Final RF, SHAP (2 min), susceptibility map, validation map, rate curves | ~6 min |

The whole notebook takes about 43 minutes. More than half of that is step 6, and the largest part of step 6 is the 1:3 run, because the SVM's training time grows quickly with the number of points. Cells 5.4, 6.2–6.4, 7.1, 7.2, 7.5 and 7.7 print their own run times.

---

## What each step produces

| Cell | Output (in `outputs/`) |
|---|---|
| 2.6, 2.8 | `figures/fig_inventory_periods.png`, `landslides_clean.gpkg` |
| 3.4, 3.8, 3.9 | `figures/fig_terrain.png`, `figures/fig_samples.png`, `features.csv` (11,765 points × 20 columns) |
| 4.2–4.6 | `table_univariate.csv`, `figures/fig_distributions.png`, `fig_geology_share.png`, `fig_correlation.png`, `fig_pca.png` |
| 5.4–5.8 | `oof_modelA.csv`, `best_params_modelA.csv`, `table_results_modelA.csv`, `figures/fig_roc_pr_modelA.png`, `fig_confusion_modelA.png` |
| 6.2–6.9 | `oof_modelA_noshake.csv`, `oof_modelA_1to3.csv`, `oof_modelB.csv`, `table_modelA_vs_B.csv`, `table_bahrabise.csv`, `table_bahrabise_diagnostic.csv`, `table_experiments.csv`, `figures/fig_modelA_vs_B.png` |
| 7.2–7.8 | `table_shap_importance.csv`, `susceptibility_probability.tif`, `susceptibility_class.tif`, `table_susceptibility_classes.csv`, `table_rate_curves.csv`, `figures/fig_shap_*.png`, `fig_susceptibility_map.png`, `fig_class_shares.png`, `fig_rate_curves.png` |

Both GeoTIFFs open in QGIS or ArcGIS (EPSG:32645). The class map carries its own color table: 1 Low, 2 Moderate, 3 High, 4 Very High.

---

## Method in brief

**Labels.** The inventory doesn't record when each slope failed. Its `Name` field holds the date of the Google Earth image the slide was traced on, so a slide first seen between 25 April and 30 June 2015 is treated as co-seismic (2,477 slides) and one seen before 25 April 2015 as pre-earthquake (1,857). The 19 records with unreadable dates and the 394 slides first seen after June 2015 are left out.

**Samples.** Each landslide becomes one point at its polygon centroid. Non-landslide points are random locations on valid DEM inside the study area (the convex hull of all slides plus 2 km), at least 150 m from any mapped landslide of any date. Three are drawn per co-seismic slide: the first third forms the main 1:1 set, and all of them together form the 1:3 set.

**Features.** Elevation, slope, aspect (as sine and cosine), plan and profile curvature and local relief come from the DEM, reprojected to UTM 45N at 90 m. Main-shock and aftershock PGA are interpolated bilinearly from the ShakeMap grids, and PGA × sin(slope) combines the two strongest signals into one feature (single-feature AUC 0.834). Geology enters as a class, with rare and undocumented classes grouped as "Other". Peak ground velocity and the larger of the two PGAs were dropped as redundant with main-shock PGA.

**Validation.** A random train/test split would let a model learn a valley and then be tested on the slope next to it, so every score here comes from nested leave-one-district-out cross-validation instead. The five districts with fewer than 25 co-seismic slides are merged into one group, which gives nine outer folds. Inside each training fold a 3-fold, district-grouped grid search picks the hyperparameters on ROC-AUC, and scaling, one-hot encoding and the Frequency Ratio classes are all fitted there too, never on the held-out district.

**Map.** The final Random Forest scores all 2,612,676 valid 90 m cells. Classes are cut by share of area: Low 40%, Moderate 30%, High 20%, Very High 10%. The model fits its own training points almost perfectly (99% of them land in Very High), so the map is validated on a second version trained without Sindhupalchok, not on the landslides it learned from.

---

## Reproducibility

Every random step uses `SEED = 42` (cell 1.3), and a second run on the same machine reproduced every reported number exactly. Tested versions: Python 3.14.8, numpy 2.5.4, pandas 3.0.6, geopandas 1.2.0, shapely 2.2.0, pyproj 3.8.0, rasterio 1.5.2, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, shap 0.53.0. Cell 1.1 prints the versions actually in use. Other versions should run, but they haven't been tested, and small numerical differences are possible.

---

## Limitations

The most serious limit is the one the Bahrabise test exposed: the model learned what ICIMOD's mappers recorded as a landslide, and an inventory mapped at a finer scale with different conventions doesn't fully agree with it. A score on one inventory says little about how the model would rate another.

The rest are smaller. Image dates aren't failure dates, so the co-seismic set may include a few slides with other causes. The median landslide (about 3,500 m²) is smaller than one 90 m cell (8,100 m²), so small failures are described by coarse terrain. ShakeMap PGA is mostly modeled rather than measured. Non-landslide points take the district of their nearest landslide, because no district boundary layer was used. The model has been tested on one earthquake only.

The map ranks relative susceptibility under this one earthquake's shaking pattern. It's a research result, not an official hazard map, and a Low class doesn't mean a slope is safe.
