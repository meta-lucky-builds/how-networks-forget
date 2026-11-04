## cifar_class: lr 0.01, epochs 20, m* 1, n 2, w0 acc 0.688, gates passed True
H6a probes overstate: naive 0.552, S_readout 0.146, gap 0.406 [0.3505936773678935, 0.4688550832151674], fixed S 0.196
readout acc A 0.645 to 0.357, new 0.286 to 0.304
H6b mostly unrecovered: access 0.15 [0.07002303160808261, 0.21467280363295896], recoverable 0.39 [0.33877604457936167, 0.4411398650823565]; fixed access 0.20, recoverable 0.39
gates: sham_readout -0.015, total_erasure_readout 0.035, sham_finetune_low -0.034, total_erasure_finetune_low 0.009, planted_access_readout 1.000, planted_access_none 0.124, matched_gap 0.960
lag/budget | S_none | access same, fixed | unrecovered same, fixed | top | naive, gap | S by key
1/2 | 0.00 | 0.15, 0.20 | 0.61, 0.61 | finetune_low | 0.55, 0.41 | mask=0.20 none=0.00 bn_recal=0.23 readout=0.15 readout_early=0.26 finetune_low=0.39 restore_head=0.27 restore_body=0.97
2/2 | 0.00 | 0.07, 0.15 | 0.82, 0.71 | finetune_low | 0.53, 0.46 | mask=0.08 none=0.00 bn_recal=0.07 readout=0.07 readout_early=0.09 finetune_low=0.18 restore_head=0.10 restore_body=0.97
3/2 | 0.00 | 0.07, 0.17 | 0.88, 0.63 | readout_early | 0.54, 0.47 | mask=0.05 none=0.00 bn_recal=0.04 readout=0.07 readout_early=0.12 finetune_low=0.11 restore_head=0.05 restore_body=0.97
5/2 | 0.00 | -0.01, 0.09 | 0.95, 0.63 | readout_early | 0.49, 0.50 | mask=0.01 none=0.00 bn_recal=0.00 readout=-0.01 readout_early=0.05 finetune_low=0.04 restore_head=0.02 restore_body=0.97
## cifar_task: lr 0.01, epochs 10, m* 2, n 2, w0 acc 0.671, gates passed True
H6a probes overstate: naive 0.594, S_readout 0.038, gap 0.556 [0.4644657900696704, 0.6741041113550765], fixed S 0.184
readout acc A 0.558 to 0.332, new 0.281 to 0.321
H6b mostly unrecovered: access -0.66 [-0.8846434029022902, -0.4675311642781979], recoverable 0.20 [0.1279466941100098, 0.2630069472877869]; fixed access -0.42, recoverable 0.20
gates: sham_readout -0.067, total_erasure_readout 0.048, sham_finetune_low -0.006, total_erasure_finetune_low 0.003, planted_access_readout 1.000, planted_access_none 0.416, matched_gap 0.460
lag/budget | S_none | access same, fixed | unrecovered same, fixed | top | naive, gap | S by key
1/2 | 0.61 | -0.82, -0.69 | 0.80, 0.85 | finetune_low | 0.67, 0.39 | none=0.61 bn_recal=0.60 readout=0.29 readout_early=0.35 finetune_low=0.68 restore_head=0.61 restore_body=1.00
2/2 | 0.42 | -0.66, -0.42 | 0.80, 0.80 | finetune_low | 0.59, 0.56 | none=0.42 bn_recal=0.41 readout=0.04 readout_early=0.11 finetune_low=0.54 restore_head=0.42 restore_body=1.00
3/2 | 0.31 | -0.25, -0.05 | 0.81, 0.72 | finetune_low | 0.64, 0.50 | none=0.31 bn_recal=0.31 readout=0.14 readout_early=0.22 finetune_low=0.44 restore_head=0.31 restore_body=1.00
5/2 | 0.20 | -0.23, 0.02 | 0.86, 0.53 | finetune_low | 0.61, 0.59 | none=0.20 bn_recal=0.21 readout=0.02 readout_early=0.19 finetune_low=0.32 restore_head=0.20 restore_body=1.00
