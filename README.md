# EEG-Controlled Javelin Throw

This repository combines a low-cost EEG front end with a simple Pygame demonstration. The focus is the EEG system itself: electrode pickup, analog conditioning, noise rejection, and signal interpretation.

The project is informed by Ronan Byrne’s report, *Development of a Low Cost, Open-source, Electroencephalograph-Based Brain-Computer Interface*, which describes a battery-powered, single-channel EEG measurement system built around protection, instrumentation amplification, gain, anti-alias filtering, driven-right-leg feedback, and optoisolation.

## EEG System

The schematic photo included in this workspace shows a compact EEG setup with three electrode leads feeding a breadboarded analog front end and an Arduino Uno for interface and data transfer. The layout reflects the same low-cost, open-source design philosophy as the paper: keep the acquisition chain simple, isolate the user, and extract usable brain activity with minimal hardware.

The original work emphasizes:

- Single-channel EEG measurement for a low-cost build
- Battery-powered operation for improved user safety
- A protection stage to limit input risk
- An instrumentation amplifier for weak biopotential signals
- Gain staging to lift the EEG into a usable range
- An anti-alias filter before digitisation
- Driven-right-leg feedback to reduce common-mode interference
- Optoisolation between the measurement circuit and the computer-facing side
- Low-cost electrode cups to reduce the overall build cost

## What The EEG Measures

EEG is a very small biopotential signal, so the dominant challenge is not just measurement but clean measurement. The paper and schematic both point toward the same practical concerns:

- The signal amplitude is extremely low compared with environmental noise
- Electrode contact quality matters as much as the amplifier chain
- Movement, mains interference, and poor grounding can easily dominate the recording
- Filtering and reference design are essential before any interpretation step

In the report, the system is validated using occipital-lobe electrode placement to measure alpha activity while the user opens and closes their eyes. That makes the core EEG concept easy to see: brain state changes can appear as measurable frequency changes when the electrodes and front end are built correctly.

The same report also uses steady-state visually evoked potential testing, where a user looks at flashing checker stimuli and the FFT magnitude at the target stimulation frequencies is compared against no-stimulus data. This is a practical example of how EEG can be turned into a feature extraction problem rather than a direct waveform-reading problem.

## Signal Interpretation

The report’s approach is grounded in frequency-domain analysis.

Instead of relying on a raw voltage trace, the EEG is examined through FFT magnitude at relevant frequencies. That approach is important because many useful EEG effects are easier to separate in the spectral domain than in the time domain.

Key ideas from the paper include:

- Recording long runs of no-stimulus and stimulus EEG data
- Comparing FFT magnitude at target frequencies
- Building statistical models from those distributions
- Using separation between conditions to support classification

This makes the project a good example of low-cost EEG done as signal processing, not just hardware assembly.

## Hardware Notes

If you are working with a similar build, the critical setup factors are:

- Stable electrode contact and consistent placement
- A quiet power environment
- Good shielding and grounding practice
- Correct reference and bias handling
- Careful separation of the analog front end from the USB-connected computer side

The schematic image in this workspace is useful as a reference for how compact the acquisition chain can be when the goal is educational EEG experimentation rather than clinical-grade recording.

## This Repository

This repository uses the EEG signal as an input to a simple demo application. The software here is secondary to the hardware idea: it gives the EEG signal a visible outcome so the acquisition and calibration process can be tested.

## Javelin Demo

The javelin game is a lightweight visual demonstration. After calibration, the measured EEG-derived value influences throw strength, and the result is shown as a single throw distance in a physics-based scene.

## Credit

Credit for the original low-cost open-source EEG BCI work goes to:

Ronan Byrne, *Development of a Low Cost, Open-source, Electroencephalograph-Based Brain-Computer Interface*

GitHub repository: [RonanB96/Low-Cost-EEG-Based-BCI](https://github.com/RonanB96/Low-Cost-EEG-Based-BCI)

This repository is inspired by that report and its open, practical approach to EEG acquisition.