# Modeled Scenarios

## Scenario 0
Setup consists of omni-directional speaker and microphone, and the model trains on the raw binaural signal

### Binaural Data
- Listener is modeled as 2 omnidirectional microphones
- Listener ears are 18 cm apart
- Speaker distance is 2.5 m
- Speaker is a generated chirp (same signal for all simulations)

### Model
- Model trains on the pure binaural signals

### Results
The most important result for this simple model, is that it has an expected front/back mismatch. Due to how simplistic the model is it is able not able to tell slight angle changes apart from each other, so it seems to choose either left or right as the source direction.

![Loss Graph](Scenario%20Results\scenario_0\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_0\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_0\abs_angular_error.png)


## Scenario 1
The setup is the same as scenario 0, but the input of the model will be preprocessed handcrafted features.

### Binaural Data
- Listener is modeled as 2 omnidirectional microphones
- Listener ears are 18 cm apart
- Speaker distance is 2.5 m
- Speaker is a generated chirp (same signal for all simulations)

### Model
- Preprocess handcrafted features (ITD, ILD, Interaural Cross-Correlation)
- Model trains on handcrafted features only

### Results
We can see that this models outputs the same types of error, meaning that the simple dataset does not hold enough information for the model to estimate orientation. Or that the CNN in scenario 0 is able to find the same features from the raw binaural signal as the MLP in scenario 1 is trained on.

![Loss Graph](Scenario%20Results\scenario_1\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_1\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_1\abs_angular_error.png)


## Scenario 2
this is not the true scenario 2, as that would be a fusion model trained on the raw signal.

The setup is the same as scenario 0, but the input of the model will be combined of handcrafted features and raw binaural input.

### Binaural Data
- Listener is modeled as a singular HRTF
- Speaker distance is 2.5 m
- Generate multiple chirps due to one HRTF only having 48 measurements in the horizontal plane

### Model
- Preprocess frequency-dependent features (ITD, ILD, Interaural Cross-Correlation)
- Model trains on the combined features and raw binaural input

### Results
Since we have added an HRTF to the listener, we can now compute frequency-dependent features. This takes longer for the model to train on but the predictions ends up being more accurate.

![Loss Graph](Scenario%20Results\scenario_2\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_2\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_2\abs_angular_error.png)


## Scenario 3

### Binaural Data
- Listeners correspond to multiple HRTFs
- Speaker distance is 2.5 m
- Generate multiple chirps to emit at all horizontal-plane measurement positions(48)

### Model
- Preprocess frequency-dependent features (ITD, ILD, Interaural Cross-Correlation)
- Model trains on the combined features and raw binaural input

### Results
We have a lot of samples, so training takes longer(actually it is feature extraction that takes a lot of time), but the results are very good.

![Loss Graph](Scenario%20Results\scenario_3\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_3\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_3\abs_angular_error.png)


## Scenario 5/6/7(?)

### Binaural Data
- Talker corresponds to a singular VDP
- Receiver distance is at 1 m
- Generate multiple chirps to emit towards all horizontal-plane measurement positions(60)

### Model
- Preprocess frequency-dependent features, for a monaural signal (Band dB, log RMS)
- Model trained only on preprocessed features

### Results
As we have a monaural signal, we have the same failure points as we had in the beginning with the omnidirectional ears. we are unable to tell left/right apart from each other.

![Loss Graph](Scenario%20Results\scenario_5\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_5\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_5\abs_angular_error.png)

When we instead trained the Fusion Model, that is also using the raw signal, we can see that most of the errors are gone:

![Loss Graph](Scenario%20Results\scenario_6\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_6\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_6\abs_angular_error.png)

When we further train the Fusion Model on a dataset of mutiple talkers, the previous error is more pronounced:

![Loss Graph](Scenario%20Results\scenario_7\loss_graph.png)

![Angular prediction Error](Scenario%20Results\scenario_7\angular_prediction_error.png)

![Absolute Angular Error](Scenario%20Results\scenario_7\abs_angular_error.png)

## Scenario X

### Binaural Data

### Model

### Results


# Types of Data
## Listener Focused
0. Simple (omnidirectional ears)
1. Single HRTF
2. Multiple HRTF

## Speaker Focused
3. Simple (directional signal) (does this even make sense in free-field)
4. Single VDP
5. Multiple VDP

## Jointly Focused
6. Single HRTF + Single VDP
7. Single HRTF + Multiple VDP
8. Multiple HRTF + Single VDP
9. Multiple HRTF + Multiple VDP

## Extras
10. Room Acoustics (Reverberation)
11. Added noise