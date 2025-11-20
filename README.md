# ML-Inspired-Microwave-Imaging-for-phantom-based-tumor-Detection
Overview:
This project implements a complete microwave imaging pipeline for tumor vs no-tumor classification using CST simulations, numerical phantoms, and two custom-designed antennas (one transmitter and one receiver). We reconstructed 2D microwave images from CST field data and further generated 1200 synthetic images with ground-truth tumor masks to expand the dataset. Machine learning models — Random Forest and a Convolutional Neural Network (CNN) — were trained to perform tumor classification using reconstructed and synthetic images.

Features:
Numerical phantom generation
CST-based simulation workflow
Custom Tx/Rx antenna design
2D microwave image reconstruction
1200 synthetic images + tumor masks
Random Forest feature-based classifier
CNN deep-learning classifier
Modular & reproducible pipeline
