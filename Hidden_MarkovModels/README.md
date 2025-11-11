# Brain MRI Segmentation with HMRF-EM (WM/GM/CSF)

Hidden-Markov/Markov-Random-Field segmentation of brain MRI (DICOM) into **white matter, gray matter, and CSF**.  
Pipeline: robust normalization → denoise → bias-field correction → skull strip → **HMRF-EM** (Gaussian emissions + Potts spatial prior) → labels & QA.

---

## Results at a Glance
- Outputs: `segmentation_labels.npy`, `brain_mask.npy`, `class_means.npy`, `class_vars.npy`
- Optional: `segmentation_labels.nii.gz`, `brain_mask.nii.gz`, `bg_volume.npy`, QA PNGs, `report.pdf`
- Viewer: `view_labels.py` (WM=red, GM=green, CSF=blue; overlay on grayscale; axial/coronal/sagittal)

---

## Environment (Conda, Windows/Linux/macOS)

```bash
conda create -n hmm-mri python=3.10 -y
conda activate hmm-mri
# Core deps
pip install numpy<2 scipy scikit-image scikit-learn matplotlib nibabel pypdf2 threadpoolctl
# DICOM decoding (JPEG Lossless)
pip install pydicom pylibjpeg pylibjpeg-libjpeg
# (Windows alt) conda install -c conda-forge gdcm
