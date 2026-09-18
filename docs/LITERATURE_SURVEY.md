# Literature Survey — Power System Protection using ML

**Project:** Power System Protection using Machine Learning (transmission-line overcurrent protection, IDMT relay emulation, ML fault detection/classification/location, and ML-based relay coordination)

All papers below were located and verified through Crossref / Semantic Scholar / OpenAlex / publisher records (DOIs given in the References). Summaries are based on the papers' published abstracts and, where open access, full texts.

---

## 1. Fault Detection & Classification

**[1] Fault detection and classification in electrical power transmission system using artificial neural network** — M. Jamil, S. K. Sharma, R. Singh, *SpringerPlus*, 2015.
Uses three-phase voltages and currents measured at one end of a 300 km, 50 Hz transmission line (modeled in MATLAB/Simulink with the SimPowerSystems toolbox, sources at both ends) as inputs to feed-forward back-propagation ANNs (Levenberg–Marquardt training) for fault detection and fault-type classification. Faults are simulated at different locations along the line with different fault resistances and short-circuit MVA levels, and the number of hidden layers is varied to justify the network choice. The confusion matrices reported show 100% accuracy in fault detection, with satisfactory classification of all ten shunt fault types. This is one of the most-cited baseline ANN protection papers (260+ citations) and its Simulink-based dataset-generation methodology is directly relevant to our project.

**[2] An Overview of Transmission Line Protection by Artificial Neural Network: Fault Detection, Fault Classification, Fault Location, and Fault Direction Discrimination** — A. Yadav, Y. Dash, *Advances in Artificial Neural Systems (Hindawi)*, 2014.
A comprehensive review of ANN-based transmission-line protection covering fault detection, classification, location, faulted-phase selection and direction discrimination, surveying essentially all significant contributions reported up to mid-2014. It establishes the standard pipeline used in this field: simulate faults, extract voltage/current features, train an offline ANN, and deploy it as an intelligent relay. Useful as the framing reference for the whole survey.

**[3] Decision tree-based fault zone identification and fault classification in flexible AC transmissions-based transmission line** — S. R. Samantaray, *IET Generation, Transmission & Distribution*, 2009.
Applies decision trees (DTs) to lines compensated by TCSC/UPFC (FACTS devices), where conventional distance relaying is unreliable. One cycle of post-fault current and voltage samples (plus zero-sequence quantities) from fault inception is used to identify whether the fault lies before or after the FACTS device and to classify all ten shunt fault types. Tested on simulated fault data with wide variations of operating parameters, including noise, the DT approach reliably identifies fault zone and type in a large power network — an early demonstration that simple, interpretable tree models suffice for protection decisions.

**[4] Fault Diagnosis in Power Transmission Line using Decision Tree and Random Forest Classifier** — S. Lahiri, A. Chakravarty, A. De (authors as indexed), *IEEE 6th Int. Conf. on Condition Assessment Techniques in Electrical Systems (CATCON)*, 2022.
Proposes DT and Random Forest classifiers that output an encoded 13-bit binary number simultaneously identifying fault type, faulted phase and fault location on a 100 km radial transmission line. The line is modeled in MATLAB/Simulink and the classifiers are implemented in Python — the same Simulink-to-Python workflow our project adopts. Reported fault-recognition accuracy is 95–100% across different fault scenarios.

**[5] Deep learning techniques for transmission line fault classification — A comparative study** — P. K. Shukla, K. Deepa, *Ain Shams Engineering Journal (Elsevier)*, 2024.
Compares deep-learning approaches — ANN and LSTM, with and without window regression — for classifying five categories of short-circuit faults (L-G, L-L, L-L-G, L-L-L, L-L-L-G). The key argument is that deep models perform automatic feature extraction, removing the need for the sophisticated mathematical modeling and hand-crafted signal-processing (wavelet/Fourier) stages of traditional schemes. The comparative results show sequence models (LSTM-based) improving on plain ANN classification of raw waveform windows.

**[6] Deep learning through LSTM classification and regression for transmission line fault detection, diagnosis and location in large-scale multi-machine power systems** — S. Belagoune, N. Bali, A. Bakdi, B. Baadji, K. Atif, *Measurement (Elsevier)*, 2021.
Introduces three deep recurrent neural network (LSTM) models for Fault Region Identification (FRI), Fault Type Classification (FTC) and Fault Location Prediction (FLP) in a multi-machine system. Current and voltage signals from PMUs at different terminals over pre- and post-fault cycles form high-dimensional spatiotemporal input sequences. Tested on the Kundur two-area four-machine benchmark with faults of different types at various locations, the LSTM models achieve highly accurate classification and regression, and the paper (260+ citations) is the standard reference for treating protection as a sequence-learning problem.

**[7] Fault Classification and Precise Fault Location Detection in 400 kV High-Voltage Power Transmission Lines Using Machine Learning Algorithms** — Ö. Özdemir, R. Köker, N. Pamuk, *Processes (MDPI)*, 2025.
Builds a computed dataset for 400 kV mixed-conductor lines including variations in arc (fault) resistance and bus short-circuit power, then benchmarks many ML algorithms in a staged (hierarchical) structure: detect fault → identify faulted phase → locate fault. An ensemble bagged-trees algorithm achieves 100% fault detection, a neural network achieves 99.97% faulty-phase identification, and linear-regression variants give the best location results (MAE down to 0.0066 for phase-to-phase faults). Demonstrates that a cascaded detector→classifier→locator design, like ours, outperforms single monolithic models.

---

## 2. Fault Location

**[8] Support vector machine based fault classification and location of a long transmission line** — P. Ray, D. P. Mishra, *Engineering Science and Technology, an International Journal (Elsevier)*, 2016.
SVM-based fault-type and fault-distance estimation using only one post-fault cycle of current waveform. Signals are pre-processed by wavelet packet transform; energy and entropy of the decomposed coefficients form the feature matrix, refined by forward feature selection; SVM hyperparameters are optimized by particle swarm optimization (PSO). The scheme is validated on a 400 kV, 300 km line (sources at both ends) simulated with all 10 short-circuit fault types while varying fault resistance, inception angle and distance — including faults very close to either end. The PSO-tuned SVM clearly outperforms the un-tuned SVM; this paper's simulation matrix (fault type × resistance × inception angle × location on a 400 kV line) is the template our dataset generation follows.

**[9] An improved fault detection classification and location scheme based on wavelet transform and artificial neural network for six phase transmission line using single end data only** — E. Koley, K. Verma, S. Ghosh, *SpringerPlus*, 2015.
A hybrid discrete-wavelet-transform + modular-ANN fault detector, classifier and locator for the Springdale–McCalmont 138 kV, 60 Hz, 68 km six-phase line (Allegheny Power System), modeled in MATLAB with sources at both ends and 250 MW + 100 MVAr loads. All 120 possible shunt fault types are simulated with variation in fault location, fault resistance and inception angle, plus source short-circuit capacity, X/R ratio, voltage, frequency and CT saturation. The scheme detects and classifies every shunt fault within one cycle of inception and locates faults with a maximum error of ±0.688%, using single-end data only — a strong precedent for single-end ML relays.

**[10] Transmission line fault location using traveling wave frequencies and extreme learning machine** — D. Akmaz, M. S. Mamiş, M. Arkan, M. E. Tağluk, *Electric Power Systems Research (Elsevier)*, 2018.
Estimates fault distance from the natural frequencies of fault-generated traveling waves, extracted from transient records and fed to an extreme learning machine (ELM) regressor. The ELM's one-shot least-squares training makes it far faster to train than back-propagation ANNs while achieving low location errors across fault positions in the simulated test system. Represents the signal-physics + fast-ML hybrid alternative to purely data-driven locators (95+ citations).

**[11] Fault classification and localization in power transmission line based on machine learning and combined CNN-LSTM models** — N. Q. Minh, N. T. Khiem, V. H. Giang, *Energy Reports (Elsevier)*, 2024.
Simulates the IEEE 9-bus system in MATLAB/Simulink, generating more than 300,000 fault samples with varied fault type, load level, fault resistance and fault location. SVM, decision tree, logistic regression, XGBoost and ANN are compared for classification, while CNN, LSTM and a combined CNN-LSTM are used for localization. XGBoost attains 99.82% classification accuracy, and the CNN-LSTM locates faults with MAPE < 1% and MAE < 0.16 km. The scale of its Simulink-generated dataset and its classifier/locator split make it the closest recent analogue to our proposed pipeline.

---

## 3. Relay Coordination & IDMT with ML

**[12] Optimal coordination of directional overcurrent relays in interconnected power systems** — A. J. Urdaneta, R. Nadira, L. G. Pérez Jiménez, *IEEE Transactions on Power Delivery*, 1988.
The foundational formulation of directional overcurrent relay (DOCR) coordination as a parameter-optimization problem: choose time-dial and pickup settings to minimize operating times subject to primary/backup coordination-time-interval constraints, across multiple network configurations. Solutions via direct optimization and decomposition are demonstrated on systems of up to 30 buses. Every later metaheuristic or ML coordination scheme (including ours) optimizes essentially this objective.

**[13] Per-Phase and 3-Phase Optimal Coordination of Directional Overcurrent Relays Using Genetic Algorithm** — R. C. Matthews, T. R. Patel, A. K. Summers, M. J. Reno, S. Hossain-McKenzie, *Energies (MDPI)*, 2021.
Formulates DOCR coordination under high DER penetration as a mixed-integer nonlinear program solved by a genetic algorithm in MATLAB, and — unusually — treats the time-overcurrent characteristic (curve type) itself as a discrete decision variable alongside the time-dial setting, exploiting per-phase settable digital relays (SEL-751). Comparing three-phase versus per-phase settings shows the per-phase GA solution reduces relay operating times in unbalanced systems. Bridges classic IDMT curve selection and modern optimization, exactly the TMS/pickup/curve search space our ML coordination stage explores.

**[14] Optimal Coordination of Directional Overcurrent Relays Using Hybrid Firefly–Genetic Algorithm** — T. Foqha, M. Khammash, S. Alsadi, O. Omari, S. S. Refaat, K. E. Alqawasmi, A. Elrashidi, *Energies (MDPI)*, 2023.
Solves the highly constrained nonlinear DOCR coordination problem — minimizing total relay operating time over time-multiplier settings (TMS) and plug settings (PS) — with a hybrid of a modified firefly algorithm (improved brightness/distance updates and controlled randomization for fast convergence) whose solution seeds a genetic algorithm. Tested on IEEE 3-, 8- and 9-bus benchmark systems, the hybrid achieves lower total operating times than either algorithm alone and than several published metaheuristics, illustrating the state of the art in metaheuristic TMS/pickup optimization.

**[15] Overcurrent relay modeling using artificial neural network** — M. Thoeurn, A. Priyadi, A. Tjahjono, M. H. Purnomo, *International Electrical Engineering Congress (iEECON), IEEE*, 2017.
Directly relevant to IDMT emulation: a Bayesian Regularization Back-Propagation Neural Network (BRBPNN) is trained on (pickup-multiple, tripping-time) pairs to model overcurrent relay time–current characteristics, including nonconventional (non-IEC) curves that can be shaped to load damage curves and inrush behavior. Errors between the trained network, simulation, and a hardware prototype are reported as highly acceptable, and the authors note the model is intended for use inside adaptive relay coordination — i.e., an ANN standing in for the IDMT equation, which is precisely the "IDMT emulation via ML" element of our project.

**[16] Prediction of Relay Settings in an Adaptive Protection System** — A. Summers, T. Patel, R. Matthews, M. J. Reno, *IEEE PES Innovative Smart Grid Technologies Conference (ISGT)*, 2022.
Addresses what happens when the communication link of a centralized adaptive protection system fails: each relay runs a local ML model (Facebook's Prophet time-series algorithm) that predicts its own time-dial setting (TDS) and pickup current. Training/testing data are generated from a modified IEEE 123-node feeder. The models predict pickup current with average MAPE-based accuracy of 99.961% and TDS with 94.32%, judged sufficient for protection parameter prediction — direct evidence that ML can learn relay settings, the core of our "which relay should operate, with what setting" stage.

---

## 4. Adaptive Protection / Smart Grid

**[17] A Simple Adaptive Overcurrent Protection of Distribution Systems With Distributed Generation** — P. Mahat, Z. Chen, B. Bak-Jensen, C. L. Bak, *IEEE Transactions on Smart Grid*, 2011.
The reference adaptive-overcurrent scheme for systems with DG (440+ citations): because fault currents differ drastically between grid-connected and islanded operation, the relays' inverse time-overcurrent trip characteristics are updated online using only local information, after detecting the operating state (grid-connected vs. islanded) and the faulted section — the faulted section itself being identified from the relays' time-overcurrent characteristics. Simulations show correct state/section identification and faster fault clearance after the settings update. Motivates why static IDMT settings are insufficient and adaptation (rule-based here, ML in our project) is needed.

**[18] Deep Learning Based Relay for Online Fault Detection, Classification, and Fault Location in a Grid-Connected Microgrid** — B. Roy, S. Adhikari, S. Datta, K. J. Devi, A. D. Devi, F. Alsaif, S. Alsulamy, T. S. Ustun, *IEEE Access*, 2023.
Proposes an LSTM network for online fault detection and classification in a grid-connected microgrid, and an LSTM + feed-forward-ANN (back-propagation) combination for fault location. The system is simulated extensively in MATLAB/Simulink with different fault types and parameters, then validated in real time on an OPAL-RT digital simulator, with the deep models outperforming a conventional ANN baseline. This is the fullest published example of an ML model acting as the relay itself — detection, classification and location integrated in one protective device — though it stops short of coordinating multiple relays.

---

## 5. Research Gap Identified

Synthesizing the surveyed literature:

1. **Fragmentation of the pipeline.** Papers [1]–[7] address detection/classification, [8]–[11] address location, and [12]–[16] address coordination and setting optimization — but almost always in isolation. Even the most integrated works ([18], and the classifier+locator of [11]) do not close the loop from fault classification to a relay-operation decision; conversely, the coordination literature ([12]–[14]) assumes the fault has already been detected and characterized by conventional means.

2. **IDMT characteristics and ML rarely meet.** ML-based IDMT curve emulation exists ([15]) and ML-based setting prediction exists ([16]), but no surveyed paper trains an ML model to *reproduce and generalize* the IEC inverse-time behavior (operating time as a function of PSM/TMS) inside the same framework that also performs fault detection and classification.

3. **Relay selection is not framed as a learning problem.** Adaptive schemes ([16], [17]) update settings, and metaheuristics ([12]–[14]) precompute optimal TMS/pickup tables, but the discrete decision "which relay should operate for this fault" is nowhere posed as a supervised ML classification task trained on simulated fault data.

4. **Simulation methodology is proven but siloed.** The strongest works generate large labeled datasets from MATLAB/Simulink or equivalent models — 400 kV/300 km with fault-resistance and inception-angle sweeps [1], [8]; 138 kV/68 km with 120 fault types and CT-saturation/X-R variations [9]; IEEE 9-bus with >300k samples varying type, load, resistance and location [11]; IEEE 123-feeder setting data [16] — validating simulation-to-ML training as a methodology. However, each dataset serves only its single subtask.

**Gap our project fills:** an *integrated* pipeline — power-system fault simulation (transmission line modeled at a defined voltage level with swept fault types, locations, resistances and inception angles) → unified labeled dataset → ML fault detection and classification → ML fault-location estimate → ML emulation of the IDMT inverse-time characteristic → ML-based relay-selection/coordination decision (which relay operates, with what effective TMS/pickup). To our knowledge from this survey, no single published work chains all of these stages on one common simulated dataset, which is the contribution of this project.

---

## References

[1] M. Jamil, S. K. Sharma, and R. Singh, "Fault detection and classification in electrical power transmission system using artificial neural network," *SpringerPlus*, vol. 4, art. 334, 2015, doi: 10.1186/s40064-015-1080-x.

[2] A. Yadav and Y. Dash, "An overview of transmission line protection by artificial neural network: Fault detection, fault classification, fault location, and fault direction discrimination," *Advances in Artificial Neural Systems*, vol. 2014, art. 230382, 2014, doi: 10.1155/2014/230382.

[3] S. R. Samantaray, "Decision tree-based fault zone identification and fault classification in flexible AC transmissions-based transmission line," *IET Generation, Transmission & Distribution*, vol. 3, no. 5, pp. 425–436, 2009, doi: 10.1049/iet-gtd.2008.0316.

[4] S. Lahiri, A. Chakravarty, and A. De, "Fault diagnosis in power transmission line using decision tree and random forest classifier," in *Proc. 2022 IEEE 6th Int. Conf. on Condition Assessment Techniques in Electrical Systems (CATCON)*, 2022, doi: 10.1109/CATCON56237.2022.10077633.

[5] P. K. Shukla and K. Deepa, "Deep learning techniques for transmission line fault classification – A comparative study," *Ain Shams Engineering Journal*, vol. 15, no. 2, art. 102427, 2024, doi: 10.1016/j.asej.2023.102427.

[6] S. Belagoune, N. Bali, A. Bakdi, B. Baadji, and K. Atif, "Deep learning through LSTM classification and regression for transmission line fault detection, diagnosis and location in large-scale multi-machine power systems," *Measurement*, vol. 177, art. 109330, 2021, doi: 10.1016/j.measurement.2021.109330.

[7] Ö. Özdemir, R. Köker, and N. Pamuk, "Fault classification and precise fault location detection in 400 kV high-voltage power transmission lines using machine learning algorithms," *Processes*, vol. 13, no. 2, art. 527, 2025, doi: 10.3390/pr13020527.

[8] P. Ray and D. P. Mishra, "Support vector machine based fault classification and location of a long transmission line," *Engineering Science and Technology, an International Journal*, vol. 19, no. 3, pp. 1368–1380, 2016, doi: 10.1016/j.jestch.2016.04.001.

[9] E. Koley, K. Verma, and S. Ghosh, "An improved fault detection classification and location scheme based on wavelet transform and artificial neural network for six phase transmission line using single end data only," *SpringerPlus*, vol. 4, art. 551, 2015, doi: 10.1186/s40064-015-1342-7.

[10] D. Akmaz, M. S. Mamiş, M. Arkan, and M. E. Tağluk, "Transmission line fault location using traveling wave frequencies and extreme learning machine," *Electric Power Systems Research*, vol. 155, pp. 1–7, 2018, doi: 10.1016/j.epsr.2017.09.019.

[11] N. Q. Minh, N. T. Khiem, and V. H. Giang, "Fault classification and localization in power transmission line based on machine learning and combined CNN-LSTM models," *Energy Reports*, vol. 12, 2024, doi: 10.1016/j.egyr.2024.11.061.

[12] A. J. Urdaneta, R. Nadira, and L. G. Pérez Jiménez, "Optimal coordination of directional overcurrent relays in interconnected power systems," *IEEE Transactions on Power Delivery*, vol. 3, no. 3, pp. 903–911, 1988, doi: 10.1109/61.193867.

[13] R. C. Matthews, T. R. Patel, A. K. Summers, M. J. Reno, and S. Hossain-McKenzie, "Per-phase and 3-phase optimal coordination of directional overcurrent relays using genetic algorithm," *Energies*, vol. 14, no. 6, art. 1699, 2021, doi: 10.3390/en14061699.

[14] T. Foqha, M. Khammash, S. Alsadi, O. Omari, S. S. Refaat, K. E. Alqawasmi, and A. Elrashidi, "Optimal coordination of directional overcurrent relays using hybrid firefly–genetic algorithm," *Energies*, vol. 16, no. 14, art. 5328, 2023, doi: 10.3390/en16145328.

[15] M. Thoeurn, A. Priyadi, A. Tjahjono, and M. H. Purnomo, "Overcurrent relay modeling using artificial neural network," in *Proc. 2017 International Electrical Engineering Congress (iEECON)*, 2017, doi: 10.1109/IEECON.2017.8075794.

[16] A. Summers, T. Patel, R. Matthews, and M. J. Reno, "Prediction of relay settings in an adaptive protection system," in *Proc. 2022 IEEE PES Innovative Smart Grid Technologies Conference (ISGT)*, 2022, doi: 10.1109/ISGT50606.2022.9817483.

[17] P. Mahat, Z. Chen, B. Bak-Jensen, and C. L. Bak, "A simple adaptive overcurrent protection of distribution systems with distributed generation," *IEEE Transactions on Smart Grid*, vol. 2, no. 3, pp. 428–437, 2011, doi: 10.1109/TSG.2011.2149550.

[18] B. Roy, S. Adhikari, S. Datta, K. J. Devi, A. D. Devi, F. Alsaif, S. Alsulamy, and T. S. Ustun, "Deep learning based relay for online fault detection, classification, and fault location in a grid-connected microgrid," *IEEE Access*, vol. 11, pp. 62674–62696, 2023, doi: 10.1109/ACCESS.2023.3285768.
