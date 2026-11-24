# Collision laws across five recordings

Each equation predicts mean velocity over the next 0.4 s. Its coefficient vector was refitted on four recordings for each held-out score. The displayed coefficients are a separate fit to the training prefixes of all five recordings.

For each row, $\widehat{\mathbf v}_i = \mathbf b_i + V\,F_i$. The base $\mathbf b_i$ combines calibrated propulsion, velocity persistence and wall motion. $V$ is one cap diameter per forecast interval. Each $\Phi_k$ is a dimensionless vector from the term dictionary. A pair term sums its weighted direction over neighbors within 253 camera pixels.

Both geometries were scored on the same 20,518 archived structures and reference equations. The 45 rows below minimize mean training-screen WAPE and dyad coefficient instability among 18,634 structures eligible in all five projected-data folds. The velocity errors use raw measured displacements. Each recording has equal weight. The structure pool uses evidence from all five recordings. The fold-selected contact and joint comparisons use proposals generated inside each training fold.

| Correction $F_i$ | WAPE | Dyad | Mean held-out R² | Worst held-out R² | RMSE (px/s) |
|---|---:|---:|---:|---:|---:|
| $0.01581\Phi_{006} +0.007156\Phi_{082} +0.4729\Phi_{108} +0.2788\Phi_{109} +15.31\Phi_{156} +0.204\Phi_{173} +0.1172\Phi_{180} -0.2029\Phi_{181} -0.1329\Phi_{183} +0.1344\lvert w_i\rvert\Phi_{020}$ | 0.8032 | 6.7425 | 0.4210 | 0.3180 | 55.34 |
| $0.8903\Phi_{025} +0.006856\Phi_{082} +0.4803\Phi_{108} +0.2887\Phi_{109} +15.72\Phi_{156} +0.2066\Phi_{173} +0.1173\Phi_{180} -0.2039\Phi_{181} -0.1376\Phi_{183} +0.4461\lvert w_i\rvert\Phi_{002}$ | 0.8051 | 6.1373 | 0.4223 | 0.3196 | 55.27 |
| $0.3701\Phi_{012} +0.00672\Phi_{082} +0.4777\Phi_{108} +0.2728\Phi_{109} +0.2867\Phi_{173} +0.078\Phi_{180} -0.193\Phi_{181} +0.166\lvert w_i\rvert\Phi_{020} +25.45\lvert w_i\rvert\Phi_{156}$ | 0.8054 | 4.9882 | 0.4153 | 0.3063 | 55.57 |
| $0.01525\Phi_{006} +0.5553\Phi_{108} +0.3217\Phi_{109} -0.1067\Phi_{111} +16.39\Phi_{156} +0.278\Phi_{173} +0.05946\Phi_{180} +0.159\lvert w_i\rvert\Phi_{020}$ | 0.8074 | 4.0309 | 0.4063 | 0.2974 | 55.87 |
| $0.01555\Phi_{006} +0.5579\Phi_{108} +0.3233\Phi_{109} -0.1083\Phi_{111} +0.2794\Phi_{173} +0.06008\Phi_{180} +0.1533\lvert w_i\rvert\Phi_{020} +24.58\lvert w_i\rvert\Phi_{156}$ | 0.8074 | 3.8180 | 0.4069 | 0.2983 | 55.84 |
| $0.3444\Phi_{012} +0.558\Phi_{108} +0.3125\Phi_{109} -0.1107\Phi_{111} +0.2856\Phi_{173} +0.05968\Phi_{180} +0.1509\lvert w_i\rvert\Phi_{020} +24.32\lvert w_i\rvert\Phi_{156}$ | 0.8078 | 3.7837 | 0.4063 | 0.2987 | 55.87 |
| $2.607\Phi_{003} +0.009912\Phi_{082} +0.2829\Phi_{101} +0.4258\Phi_{102} -0.07772\Phi_{111} -0.005393\Phi_{146} +0.1403\Phi_{150} +0.2463\Phi_{158}$ | 0.8127 | 3.7309 | 0.3971 | 0.2795 | 56.39 |
| $0.01557\Phi_{006} +0.009989\Phi_{082} +0.2842\Phi_{101} +0.4206\Phi_{102} -0.07843\Phi_{111} +0.1352\Phi_{150} +14.85\Phi_{156}$ | 0.8148 | 3.4131 | 0.3960 | 0.2790 | 56.44 |
| $2.614\Phi_{003} +0.01002\Phi_{082} +0.2866\Phi_{101} +0.4238\Phi_{102} -0.07745\Phi_{111} +0.1369\Phi_{150} +1.269\Phi_{154}$ | 0.8149 | 3.3089 | 0.3960 | 0.2789 | 56.44 |
| $2.412\Phi_{003} +0.01022\Phi_{082} +0.2715\Phi_{101} +0.4153\Phi_{102} -0.08177\Phi_{111} +0.1341\Phi_{150} +0.1419\Phi_{158}$ | 0.8155 | 3.1737 | 0.3955 | 0.2794 | 56.46 |
| $0.3993\Phi_{012} +0.5738\Phi_{108} +0.3184\Phi_{109} -0.1136\Phi_{111} +0.08253\Phi_{180} -0.1822\Phi_{183}$ | 0.8171 | 2.6342 | 0.3990 | 0.2949 | 56.26 |
| $0.3849\Phi_{012} +0.01038\Phi_{082} +0.275\Phi_{101} +0.4225\Phi_{102} -0.0816\Phi_{111} +0.111\Phi_{158}$ | 0.8195 | 2.6336 | 0.3904 | 0.2743 | 56.68 |
| $0.009955\Phi_{082} +0.2875\Phi_{101} +0.4352\Phi_{102} -0.08352\Phi_{111} +0.1244\Phi_{158} +0.5056\lvert w_i\rvert\Phi_{002}$ | 0.8197 | 2.5818 | 0.3945 | 0.2792 | 56.49 |
| $0.01437\Phi_{006} +0.004001\Phi_{082} +0.6277\Phi_{108} +0.3646\Phi_{109} +0.1234\Phi_{150} +15.81\Phi_{156}$ | 0.8206 | 2.3741 | 0.3750 | 0.2740 | 57.25 |
| $2.284\Phi_{003} +0.00385\Phi_{082} +0.6258\Phi_{108} +0.3723\Phi_{109} +0.1282\Phi_{150} +16.03\Phi_{156}$ | 0.8206 | 2.3540 | 0.3758 | 0.2762 | 57.22 |
| $0.01676\Phi_{006} +0.01017\Phi_{082} +0.2923\Phi_{101} +0.5229\Phi_{102} +14.22\Phi_{156}$ | 0.8210 | 1.7803 | 0.3846 | 0.2738 | 56.89 |
| $2.65\Phi_{003} +0.009831\Phi_{082} +0.303\Phi_{101} +0.5295\Phi_{102} +14.57\Phi_{156}$ | 0.8213 | 1.7705 | 0.3850 | 0.2724 | 56.88 |
| $2.804\Phi_{003} +0.01015\Phi_{082} +0.2961\Phi_{101} +0.5258\Phi_{102} +1.216\Phi_{154}$ | 0.8216 | 1.7556 | 0.3848 | 0.2735 | 56.89 |
| $0.009892\Phi_{082} +0.2966\Phi_{101} +0.5318\Phi_{102} +14.67\Phi_{156} +0.4824\lvert w_i\rvert\Phi_{002}$ | 0.8221 | 1.7321 | 0.3867 | 0.2779 | 56.79 |
| $0.5538\Phi_{101} +0.7369\Phi_{102} -0.2639\Phi_{109} +14.29\Phi_{156} +0.4817\lvert w_i\rvert\Phi_{002}$ | 0.8233 | 1.6933 | 0.3840 | 0.2733 | 56.90 |
| $0.3787\Phi_{002} +0.6606\Phi_{108} +0.3897\Phi_{109} -0.006469\Phi_{145} +22.77\lvert w_i\rvert\Phi_{156}$ | 0.8249 | 1.6883 | 0.3738 | 0.2705 | 57.31 |
| $2.564\Phi_{003} +0.3465\Phi_{101} +0.5267\Phi_{102} +14.08\Phi_{156}$ | 0.8250 | 1.3433 | 0.3806 | 0.2676 | 57.07 |
| $0.01615\Phi_{006} +0.3375\Phi_{101} +0.5236\Phi_{102} +21.59\lvert w_i\rvert\Phi_{156}$ | 0.8251 | 1.2681 | 0.3809 | 0.2698 | 57.05 |
| $2.569\Phi_{003} +0.3461\Phi_{101} +0.53\Phi_{102} +22.02\lvert w_i\rvert\Phi_{156}$ | 0.8251 | 1.2382 | 0.3815 | 0.2687 | 57.02 |
| $0.01627\Phi_{006} +0.6592\Phi_{108} +0.3845\Phi_{109} +14.98\Phi_{156}$ | 0.8254 | 1.1979 | 0.3727 | 0.2702 | 57.36 |
| $2.581\Phi_{003} +0.6569\Phi_{108} +0.3944\Phi_{109} +15.29\Phi_{156}$ | 0.8255 | 1.1824 | 0.3734 | 0.2691 | 57.34 |
| $0.01636\Phi_{006} +0.6621\Phi_{108} +0.386\Phi_{109} +22.81\lvert w_i\rvert\Phi_{156}$ | 0.8258 | 1.1493 | 0.3732 | 0.2713 | 57.33 |
| $2.596\Phi_{003} +0.6599\Phi_{108} +0.396\Phi_{109} +23.24\lvert w_i\rvert\Phi_{156}$ | 0.8260 | 1.1284 | 0.3739 | 0.2702 | 57.31 |
| $3.039\Phi_{003} +0.01061\Phi_{082} +0.5417\Phi_{102} +0.4807\lvert w_i\rvert\Phi_{101}$ | 0.8273 | 1.1124 | 0.3828 | 0.2701 | 56.98 |
| $19.07\Phi_{004} +0.6592\Phi_{108} +0.4144\Phi_{109} +24.78\lvert w_i\rvert\Phi_{156}$ | 0.8273 | 1.0999 | 0.3729 | 0.2667 | 57.36 |
| $0.01819\Phi_{006} +0.3325\Phi_{101} +0.5263\Phi_{102}$ | 0.8306 | 0.7727 | 0.3766 | 0.2642 | 57.24 |
| $2.888\Phi_{003} +0.3425\Phi_{101} +0.5339\Phi_{102}$ | 0.8308 | 0.7692 | 0.3771 | 0.2627 | 57.22 |
| $0.3364\Phi_{101} +0.5369\Phi_{102} +0.5226\lvert w_i\rvert\Phi_{002}$ | 0.8312 | 0.7481 | 0.3789 | 0.2685 | 57.12 |
| $0.4042\Phi_{012} +0.66\Phi_{108} +0.3751\Phi_{109}$ | 0.8318 | 0.6692 | 0.3665 | 0.2638 | 57.61 |
| $0.3549\Phi_{002} +0.6607\Phi_{108} +0.3734\Phi_{109}$ | 0.8319 | 0.6659 | 0.3660 | 0.2627 | 57.63 |
| $2.929\Phi_{003} +0.6548\Phi_{108} +0.3981\Phi_{109}$ | 0.8321 | 0.6615 | 0.3687 | 0.2641 | 57.53 |
| $0.6718\Phi_{108} +0.3929\Phi_{109} +0.5289\lvert w_i\rvert\Phi_{002}$ | 0.8331 | 0.6473 | 0.3699 | 0.2701 | 57.45 |
| $0.6623\Phi_{108} +0.4722\Phi_{109} +24.55\Phi_{156}$ | 0.8374 | 0.6258 | 0.3546 | 0.2386 | 58.19 |
| $0.02902\Phi_{017} +0.656\Phi_{108} +0.4626\Phi_{109}$ | 0.8421 | 0.6185 | 0.3521 | 0.2358 | 58.28 |
| $0.6401\Phi_{108} +0.4699\Phi_{109} +1.49\Phi_{P002}$ | 0.8446 | 0.6156 | 0.3496 | 0.2353 | 58.39 |
| $0.6115\Phi_{102} +4.895\Phi_{105}$ | 0.8448 | 0.4042 | 0.3543 | 0.2230 | 58.25 |
| $0.6599\Phi_{108} +0.4969\Phi_{109}$ | 0.8489 | 0.3228 | 0.3417 | 0.2217 | 58.73 |
| $0.5452\Phi_{002} +1.178\Phi_{108}$ | 0.8630 | 0.2895 | 0.3092 | 0.1278 | 59.89 |
| $0.1124\Phi_{005} +1.161\Phi_{108}$ | 0.8632 | 0.2757 | 0.3072 | 0.1242 | 59.97 |
| $0.6141\Phi_{109}$ | 0.8649 | 0.0841 | 0.3219 | 0.2098 | 59.56 |

## Terms on this front

The weight is evaluated before summing neighbors. The normal points from the neighbor to the focal robot. The tangent is its 90° counterclockwise rotation. Here rho is projected center distance in cap diameters, rhoc is the cutoff in cap diameters, and q is cutoff minus distance divided by 230 pixels. D_i and D_j are the positive parts of the robots’ propulsion components toward one another. T_i is the focal robot’s signed tangential propulsion component. The propulsion components are divided by V.

| Term | Family | Weight | Direction |
|---|---|---|---|
| C002 | Distance and contact state | `q` | n |
| C003 | Distance and contact state | `q^2` | n |
| C004 | Distance and contact state | `q^3` | n |
| C005 | Distance and contact state | `q/(q+q0)` | n |
| C006 | Distance and contact state | `exp(q/q0)-1` | n |
| C012 | Distance and contact state | `1/rho-1/rhoc` | n |
| C017 | Distance and contact state | `B2` | n |
| C020 | Approach and release | `c` | n |
| C025 | Approach and release | `q*c` | n |
| C082 | Heading and orientation | `h_j` | n |
| C101 | Active propulsion | `D_j` | n |
| C102 | Active propulsion | `D_i` | n |
| C105 | Active propulsion | `q*D_j` | n |
| C108 | Active propulsion | `min(D_i,D_j)` | n |
| C109 | Active propulsion | `max(D_i,D_j)` | n |
| C111 | Active propulsion | `T_i` | t |
| C145 | Simultaneous neighbors | `N_i` | n |
| C146 | Simultaneous neighbors | `N_j` | n |
| C150 | Simultaneous neighbors | `c/(1+N_i)` | n |
| C154 | Simultaneous neighbors | `pos(dot(P_j,n))` | n |
| C156 | Simultaneous neighbors | `q*pos(dot(P_j,n))` | n |
| C158 | Simultaneous neighbors | `dot(n,F_j@n)` | n |
| C173 | Wall and wall-mediated contacts | `w_i*c_w` | m |
| C180 | Free motion controls | `v_old-v_older` | As written |
| C181 | Free motion controls | `p_i` | As written |
| C183 | Free motion controls | `abs(wi)*v_old` | As written |
| P002 | Three-factor interactions | `q*c*opp` | n |

## Matched transfer errors

Contact and joint models are selected separately within each fold by validation RMSE. The shared minimum-WAPE row uses the same equation structure in every fold. Pair errors project the relative velocity of observed neighbors onto their observed normal or tangent at the forecast origin.

| Selection | Geometry | Robot R² | Pair-normal R² | Pair-tangent R² | Normal RMSE (px/s) |
|---|---|---:|---:|---:|---:|
| base | projected | 0.0834 | -1.5891 | 0.0872 | 91.40 |
| base | raw | 0.0828 | -1.5933 | 0.0866 | 91.47 |
| contact | projected | 0.4044 | 0.1097 | 0.3987 | 55.49 |
| contact | raw | 0.4043 | 0.1076 | 0.3986 | 55.56 |
| fixed_eight | projected | 0.4053 | 0.0726 | 0.4001 | 56.40 |
| fixed_eight | raw | 0.4051 | 0.0713 | 0.4002 | 56.44 |
| joint | projected | 0.4206 | 0.0996 | 0.4068 | 55.75 |
| joint | raw | 0.4202 | 0.0997 | 0.4065 | 55.74 |
| min_wape_shared | projected | 0.4210 | 0.1027 | 0.4087 | 55.65 |
| min_wape_shared | raw | 0.4206 | 0.1006 | 0.4084 | 55.72 |

The native search used one seed, two libraries, 16 individuals and 12 generations per recording holdout. The companion beam search used the same 299-column dictionary. Two geometric columns have zero variation after the hard-cap correction. All original forecast rows passing the history warmup and availability checks were retained. Raw measured cap crossings determine contact state. The observed gap also supplies apparent-overlap terms.
