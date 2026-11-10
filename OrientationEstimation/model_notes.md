# Model Notes
We have the following types of orientation estimation models that we want to compare:
- Listener
- Speaker
- Joint Speaker/Listener

To do so we will begin with a binaural signal:
- The signal will come from a simple point source.
- The signal will be captured by a two-mic 'setup', mimicing the listener's ears, resulting in a binaural signal.

For each of the models we will look at which features gives the most promising results, as we make the model more and more complex/realistic.

## Models

### Listener Orientation Estimation Model
This model will purely estimate the orientation of the listener.

#### Features
- ITD
- ILD
- Certain Spectral Features (use best from sources, or find best ourselves)
- Hand-Crafted Features
- Frequency-dependent Features

#### Loss
- Mean-Square Error, for continuous angle (Regression)
- (Binary) Cross-Entropy, for facing/not facing (Classification)


### Speaker Orientation Estimation Model
This model will purely estimate the orientation of the speaker.

#### Features

#### Loss


### Joint Speaker/Listener Orientation Estimation Model
This model will jointly estimate the orientation of both the speaker and the listener.

#### Features

#### Loss


## Data
Things that will add complexity to the binaural signal:

- Clean Signal (the simple starting point)
- Single HRTF
- Multi HRTF
- Single VDP
- Multi VDP
- Room Conditions
- Noise


## Model Types
- 1D CNN
- 2D CNN
- MLP
- Fusion Model
- Takawale and Roy (only for joint estimation)
- Fusion-Branch Model (only for joint estimation)