# Contrastive_Activation-Oracles

This project explores whether an Activation Oracle can compare two matched activation traces from the same language model under different prompt conditions and describe the resulting change in observable behavior. For each pair, the target model is run on two contexts with an identical shared suffix, producing aligned activations (H_A) and (H_B), along with their difference (H_B-H_A). The work begins by testing whether a released AO can use these paired traces zero-shot, then fine-tunes it on contrastive examples where the target is a measured directional behavior change, such as selecting a different answer, following a hint, asking for clarification, or showing no stable change. The resulting contrastive AO is evaluated with condition swaps, shuffled-trace controls, null pairs, text-only baselines, and separate single-trace AO baselines to test whether its answers depend on the internal contrast rather than merely reconstructing the prompt or predicting the final output.

Work in Progress!

***This might fail before it takes off KEKW***

> >.< sorry for gptish english
