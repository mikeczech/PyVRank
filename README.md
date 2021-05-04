# PyVRank

Software verification is an emerging discipline of engineering whose goal is to prove the absence of defects in programs.
Research efforts over the past decades have led to sophisticated techniques for software verification, each of which makes different trade-offs in terms of soundness and efficiency.
Modern tool implementations for software verification combine a variety of those techniques.
A major challenge for engineers is therefore to select the most adequate tool for a particular task at hand.

This repository contains the implementation of a machine learning-based technique for predicting rankings of software verification competitions in such a way that tools are ranked in accordance to their expected performance on a given verification task. Predicted rankings provide engineers with interesting prospects for tool selection by giving them the opportunity to trade off tool performance against other decision factors (e.g. fees for the use of tools in a service-oriented market).
From a technical perspective, the technique combines control-flow, data dependency, control dependency, and syntactic information into one graph representation of verification tasks.
Moreover, the technique predicts tool rankings using kernel methods with a kernel framework that is tailored to utilize domain-specific aspects of our graphs.

## Paper and Other Resources
* https://arxiv.org/abs/1703.00757
* https://sv-comp.sosy-lab.org/2014/
