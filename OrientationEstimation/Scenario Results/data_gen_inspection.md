# Inspection of Binaural Data Generation Pipeline
We do not want to implement the entire binaural data generation pipeline at once. That would make it impossible to know where possible errors are happening. Instead we are progressively implementing small parts of the pipeline, and inspecting each of the parts.

The steps we make are as follows:
- Simple - Omni-directional signal to omni-directional ears
- Single HRTF - Omni-directional signal to a single HRTF
- Multiple HRTFs - Omni-directional signal to multiple HRTFs
- Single VDP - Sound through a single VDP to a single HRTF
- Multiple VDPs - Sound through multiple VDPs to a single HRTF
- Multiple HRTF/VDP - Sound through multiple VDPs to multiple HRTFs
- Speech - Speech Commands through multiple VDPs to multiple HRTFs
- Room - Same as 'Speech' but subjects are placed in a room
- Further implementations depends on outcomes from the tests above

To inspect each of the tests we will extract SOTA-derived features and train/test appropriate models.
For each of the part tests we will present: 
- Description of data
- Expected outcome / Success criteria
- Results


## Tests

### Simple
We have place an omni-directional as the source and placed two omni-directional receiver 18 cm apart as the ears.
For sound we have generated multiple chirps.

#### Expected outcome / Success criteria
As there is nothing to 'shadow' between the ears, we expect to not be able to tell front and back apart. It should still be possible to tell whether the signal comes from left or right, but with no front/back knowledge, front-left and back-left is essentially seen as the same.

#### Results
We see that the model does not know front/back from each other, and therefore only predicts either left or right:

![Loss Graph](scenario_0\loss_graph.png)

![Angular prediction Error](scenario_0\angular_prediction_error.png)

![Absolute Angular Error](scenario_0\abs_angular_error.png)

---
### Single HRTF
We still have an omni-directional source, but the receivers are an actual HRTF.
Sounds are still generated chirps.

#### Expected outcome / Success criteria
We expect to now also be able to tell front and back from each other for the singular HRTF that is used.

#### Results
Here we have very little data, but the predictions follow the correct tendency:

![Loss Graph](scenario_2\loss_graph.png)

![Angular prediction Error](scenario_2\angular_prediction_error.png)

![Absolute Angular Error](scenario_2\abs_angular_error.png)

---
### Multiple HRTFs
We still have an omni-directional source, and the receivers are from multiple HRTFs.
Sounds are still generated chirps.

#### Expected outcome / Success criteria
We expect to be able to tell the orientation of even unknown listeners.

#### Results
This fits our expectations very close:

![Loss Graph](scenario_3\loss_graph.png)

![Angular prediction Error](scenario_3\angular_prediction_error.png)

![Absolute Angular Error](scenario_3\abs_angular_error.png)

---
### Single VDP
Our source is now an actual VDP, and we are only using a single HRTF.
Sounds are still generated chirps.

#### Expected outcome / Success criteria
We expect to be able to tell the orientation of the singular VDP that is used.

#### Results
Previously we trained on omni-directional ears, which looked like this:

![Loss Graph](scenario_6\loss_graph.png)

![Angular prediction Error](scenario_6\angular_prediction_error.png)

![Absolute Angular Error](scenario_6\abs_angular_error.png)

I ran this again with a singular HRTF, but apparently i now have an issue with how I combine features. I made changes to spectrogram features.

---
### Multiple VDPs
Our source is now multiple VDPs, and we are still only using a single HRTF.
Sounds are still generated chirps.

#### Expected outcome / Success criteria
We expect to be able to tell the orientation of even unknown VDPs.

#### Results
Previously we trained on omni-directional ears, which looked like this:

![Loss Graph](scenario_7\loss_graph.png)

![Angular prediction Error](scenario_7\angular_prediction_error.png)

![Absolute Angular Error](scenario_7\abs_angular_error.png)

I ran this again with a singular HRTF, but apparently i now have an issue with how I combine features. I made changes to spectrogram features.

---
### Multiple HRTFs/VDPs
We now have multiple VDPs as our sources, and our receivers are multiple HRTFs.
Sounds are stil generated chirps.

#### Expected outcome / Success criteria
We expect to be able to tell the orientation of both the listener and the talker.

#### Results
I skipped this step for now since we have multiple HRTFs and VDPs, so combining them is not a 'small' test anymore

---
### Speech
We still have multiple VDPs and HRTFs, but now the sounds comes from the speech commands dataset.

#### Expected outcome / Success criteria
We expect to still be able to tell orientation of both the listener and the talker.

#### Results
I have realized that we only have results for unknown/unknown, so the following results are bad

Ear-drum HRTF:

![Loss Graph](ED\loss_graph.png)

![Angular prediction Error](ED\combined_abs_angular_error.png)

Here it does actually look like the model is starting to learn a pattern for listeners:

![Absolute Angular Error](ED\listener_angular_prediction_error.png)
![Absolute Angular Error](ED\talker_angular_prediction_error.png)

BTE-FR HRTF:

![Loss Graph](BTE_FR\loss_graph.png)

![Angular prediction Error](BTE_FR\combined_abs_angular_error.png)

![Absolute Angular Error](BTE_FR\listener_angular_prediction_error.png)
![Absolute Angular Error](BTE_FR\talker_angular_prediction_error.png)

---
### Room
#### Expected outcome / Success criteria
#### Results

---
### Template
#### Expected outcome / Success criteria
#### Results
