# Novelty Search — Close Papers Found

**Date:** 2026-06-13  
**Purpose:** Record papers discovered during a novelty check for "2D spatially varying JCA equivalent-fluid porous medium inverse-designed for analog wave computation". The three papers marked **Included** were added to the thesis bibliography; the rest are retained here for future reference.

## Included in the thesis bibliography

1. **Pan B R, Song X, Xu J J, Sui D, Xiao H, Zhou J, Gu J.** Accelerated inverse design of customizable acoustic metaporous structures using a CNN-GA-based hybrid optimization framework. *Applied Acoustics*. 2023;210:109445. DOI:10.1016/j.apacoust.2023.109445  
   *Note:* 2D spatial variation of metaporous geometry, but objective is sound absorption and a CNN surrogate replaces the forward solve.

2. **Boulvert J, Cavalieri T, Costa-Baptista J, Schwan L, Romero-García V, Gabard G, Fotsing E R, Ross A, Mardjono J, Groby J-P.** Optimally graded porous material for broadband perfect absorption of sound. *Journal of Applied Physics*. 2019;126(17):175101. DOI:10.1063/1.5119715  
   *Note:* Continuous through-thickness gradient of JCAL parameters optimised for absorption. Closest existing use of spatially varying equivalent-fluid porous parameters, but 1D and absorption-focused.

3. **Parsa A, Wang D, O'Hern C S, Shattuck M D, Kramer-Bottiglio R, Bongard J.** Evolution of acoustic logic gates in granular metamaterials. In: *International Conference on the Applications of Evolutionary Computation (Part of EvoStar)*. Springer; 2022:93–109.  
   *Note:* Acoustic logic gates (AND/XOR) evolved in a discrete granular material. Shows inverse design for computation in a material, but substrate is particles, not continuous JCA field.

## Absorption/transmission inverse design of porous/metaporous materials

- **Cao F *et al.*** Inverse design of bending channel sound-absorbing structures with porous material by two-stage deep neural network model. *Physica Scripta*. 2025;100(5):055963. DOI:10.1088/1402-4896/adcd0d
- **Lee S Y *et al.*** Deep learning-based prediction and interpretability of physical phenomena for metaporous materials. *Materials Today Physics*. 2024;??  
  *Note:* CNN-based absorption prediction from metaporous geometry.
- **Zhang H J, Wang Y, Lu K Y, Zhao H, Yu D, Wen J.** SAP-Net: deep learning to predict sound absorption performance of metaporous materials. *Materials & Design*. 2021;212:110156. DOI:10.1016/j.matdes.2021.110156
- **Yang H T, Zhang H J, Wang Y, *et al.*** Prediction of sound absorption coefficient for metaporous materials with convolutional neural networks. *Applied Acoustics*. 2022;200:109052.
- **Guan X *et al.*** Optimization of graded porous acoustic absorbers based on triply periodic minimal surfaces. *Journal of Sound and Vibration*. 2025.  
  *Note:* Cites Boulvert 2019; 3D graded porous absorbers.
- **Su Z, Wang Q, Chen Z-G, Lu M-H.** Hybrid porous Helmholtz resonator for low-frequency broadband absorption. *Physical Review Applied*. 2024;22:044032. DOI:10.1103/PhysRevApplied.22.044032

## Acoustic logic / analog computing in non-porous substrates

- **Li Y *et al.*** Realization of acoustic tunable logic gate composed of soft materials. *Results in Physics*. 2024;??  
  *Note:* Tunable logic gates in soft chiral phononic crystals via mechanical strain.
- **Lan J *et al.*** Acoustic multifunctional logic gates and amplifier based on passive parity-time symmetry. *Physical Review Applied*. 2020;13:034047.
- **Li F, Anzel P, Yang J, Kevrekidis P G, Daraio C.** Granular acoustic switches and logic elements. *Nature Communications*. 2014;5:5311.
- **Bringuier S, Khaled M A, Chen J B, Kovačič I, Goodman E.** Acoustic logic gates based on self-collimating phononic crystals. *Journal of Applied Physics*. 2016;??  
  *Note:* Linear interference NAND/XOR/NOT in sonic crystals.

## JCA parameter identification / surrogate modelling (not design for computing)

- **Atalla Y, Panneton R.** Inverse acoustical characterization of open cell porous media using impedance tube measurements. *Journal of the Acoustical Society of America*. 2005;??
- **Santoni A, Pompoli R, Marescotti C, Fausti P.** Multi-density inversion characterisation method for fibrous material. *Forum Acusticum 2025*.
- **Van Damme B *et al.*** Enhancement of the sound absorption of closed-cell mineral foams by perforations. arXiv:2408.14933. 2024.
- **Schmid J M, Schmid J D, Marburg S.** Physics-informed neural operators for the in situ characterization of locally reacting sound absorbers. arXiv:2604.07412. 2026.

## General physical-neural-network / wave-as-RNN context (already cited)

- **Hughes T W *et al.*** Wave physics as an analog recurrent neural network. *Science Advances*. 2019;5(12):eaay6946.
- **Stein M M.** Physical neural networks using acoustics and … Cornell PhD thesis, 2024.

## Search queries used

Google Scholar (via WebBridge):
- `"JCA" "Helmholtz" "inverse design" acoustic`
- `"porous" "acoustic" "analog computing"`
- `"metaporous" "logic gate"`
- `"porous" "acoustic" "computing"`
- `"acoustic wave computing" "porous"`
- `"inverse-designed" "acoustic" "classification"`

Web search / arXiv:
- `site:arxiv.org porous acoustic analog computing`
- `site:arxiv.org "Johnson-Champoux-Allard" "logic" OR "classification"`
- `site:arxiv.org "physical neural network" acoustic porous`
- `"Applied Acoustics" 210 2023 Pan metaporous sound absorption`

## Bottom line

No published work was found that combines **all four** elements of the thesis: (1) 2D spatially varying (2) JCA equivalent-fluid rigid-frame porous medium, (3) inverse-designed by direct PDE solve + gradient-free optimisation, (4) for an analog wave-computing / classification / logic task. The closest works differ in objective (absorption), dimensionality (1D), or substrate (granular / phononic crystal / scalar wave).
