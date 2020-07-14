# Assignment 1

In this assignment, you will

* Run Hugin on a given network
* Simulate cases using Hugin
* Calculate marginal and conditional mutual information (using true probabilities and estimated probabilities from the data)
* Analyze a simulated dataset using a Jupyter notebook

Note: this is an individual assignment. Please work on it on your own. If you need help, either ask on Piaazza or see me during my office hours.

__Deadline: 10pm CST, Friday, February 15th, 2019__

1. Please clone this repository to your local computer.
2. Using Hugin, simulate 1,000 cases. Missing percentage should be set to 0, and quotes should be set to no. Save the simulated data as synthetic_v2_1000.dat.
3. Create a Jupyter Notebook called synthetic_v2.ipynb. Load synthetic_v2_1000.dat to the Jupyter Notebook. Reuse as much code as you like from the notebooks folder.
4. Edit this file (READMe.md) and answer the following questions.
5. Submit (add, commit, push as needed) the following files. Required: README.md, synthetic_v2_1000.dat, synthetic_v2.ipynb. Optional: any other file you think is necessary (for e.g., pictures of your handwritten calculations).

Please do not edit above this line.

---

__Question 1:__ 

__a)__ What is the MI(Label, F1) according to the Bayesian network structure and parameters? Fill in the following tables and provide your final answer.  Feel free to use Hugin, Python code, and calculators. For logarithm, make sure that you use log2.

P(Label, F1) 

             = P(F1| Label) P(Label)

| Label | F1 | P(Label, F1) |
| --- | --- | --- |
| A | True | (0.90)(0.50) = 0.45 |
| B | True | (0.20)(0.50) = 0.10  |
| A | False | (0.10)(0.50) = 0.05 |
| B | False | (0.80)(0.50) = 0.40 |

| Label | P(Label) |
| --- | --- |
| A | 0.50 |
| B | 0.50 |

| F1 | P(F1) |
| --- | --- |
| True | 0.55 |
| False | 0.45 |

MI(Label, F1) = Sum( P(Label, F1) log(P(Label, F1)/(P(Label)P(F1))))
              
              = P(A, True)log(P(A, True)/((P(A)P(True)))
              
                + P(B, True)log(P(B, True)/((P(B)P(True)))
                
                + P(A, False)log(P(A, False)/((P(A)P(False)))
                
                + P(B, False)log(P(B, False)/((P(B)P(False)))
              
              = 0.39731260974948646

__b)__ What is the MI(Label, F1) in the synthetic_v2_1000.dat? You can use mutual_info_classif from sklearn but remember to scale it using np.log(2), and remember to set the discrete feature settings correctly. See the notebooks/feature_importance_v1.ipynb for details.

Simulated MI(Label, F1) = 0.42399351

__Question 2:__ 

__a)__ What is the MI(Label, F2) according to the Bayesian network structure and parameters? Fill in the following tables and provide your final answer.

P(Label, F2) 
        
             = P(F2| Label) P(Label)

| Label | F2 | P(Label, F2) |
| --- | --- | --- |
| A | True | (0.40)(0.50) = 0.20 |
| B | True | (0.70)(0.50) = 0.35 |
| A | False | (0.60)(0.50) = 0.30 |
| B | False | (0.30)(0.50) = 0.15 |

| Label | P(Label) |
| --- | --- |
| A | 0.50 |
| B | 0.50 |

| F2 | P(F2) |
| --- | --- |
| True | 0.55 |
| False | 0.45 |

MI(Label, F2) = Sum( P(Label, F2) log(P(Label, F2)/(P(Label)P(F2))))

              = P(A, True)log(P(A, True)/((P(A)P(True)))
              
                + P(B, True)log(P(B, True)/((P(B)P(True)))
                
                + P(A, False)log(P(A, False)/((P(A)P(False)))
                
                + P(B, False)log(P(B, False)/((P(B)P(False)))
              
              = 0.06665370714512754

__b)__ What is the MI(Label, F2) in the synthetic_v2_1000.dat?

Simulated MI(Label, F2) = 0.0571478


__Question 3:__ 

__a)__ What is the MI(Label, F3) according to the Bayesian network structure and parameters? Fill in the following tables provide your final answer.

P(Label, F3) 
            
             = P(F3| Label) P(Label)

| Label | F3 | P(Label, F3) |
| --- | --- | --- |
| A | True | (0.86)(0.50) = 0.43 |
| B | True | (0.23)(0.50) = 0.115 |
| A | False | (0.14)(0.50) = 0.07 |
| B | False | (0.77)(0.50) = 0.385 |

| Label | P(Label) |
| --- | --- |
| A | 0.50 |
| B | 0.50 |

| F3 | P(F3) |
| --- | --- |
| True | 0.545 |
| False | 0.455 |

MI(Label, F3) 
              
              = Sum( P(Label, F3) log(P(Label, F3)/(P(Label)P(F3))))

              = P(A, True)log(P(A, True)/((P(A)P(True)))
              
                + P(B, True)log(P(B, True)/((P(B)P(True)))
                
                + P(A, False)log(P(A, False)/((P(A)P(False)))
                
                + P(B, False)log(P(B, False)/((P(B)P(False)))
                
              = 0.31302411388619616

__b)__ What is the MI(Label, F3) in the synthetic_v2_1000.dat?

Simulated MI(Label, F3) = 0.34144697


__Question 4:__ 

__a)__ What is the MI(Label, F1 \| F3 = True) according to the Bayesian network structure and parameters? Fill in the following tables provide your final answer.

P(X | Y) = P(X,Y)/P(Y)

P(Label, F1| F3=TRUE) 

                      = P(Label,F1,F3)/P(F3) 

                      = P(F3)P(F1|F3)P(Label|F1,F3)/P(F3) 
                      
                      = P(F1|F3=TRUE)P(Label|F1,F3=TRUE)

| Label | F1 | P(Label, F1 \| F3 = True) |
| --- | --- | --- |
| A | True | (0.9587)(0.8182) = 0.78440834 |
| B | True | (0.9587)(0.1818) = 0.17429166 |
| A | False | (0.0413)(0.1111) = 0.00458843 |
| B | False | (0.0413)(0.8889) = 0.03671157 |

| Label | P(Label \| F3=True) |
| --- | --- |
| A | 0.789 |
| B | 0.211 |

| F1 | P(F1 \| F3=True) |
| --- | --- |
| True | 0.9587 |
| False | 0.0413 |

MI(Label, F1 \| F3 = True) 

                 = Sum[P(Label, F1| F3 = True) log(P(Label, F1| F3 = True)/(P(Label)P(F1| F3 = True)))]

                 = P(A, True| True)log(P(A, True| True)/((P(A| True)P(True| True)))
              
                   + P(B, True| True)log(P(B, True| True)/((P(B| True)P(True| True)))
                
                   + P(A, False| True)log(P(A, False| True)/((P(A| True)P(False| True)))
             
                   + P(B, False| True)log(P(B, False| True)/((P(B| True)P(False| True)))
              
                 = 0.06686299074387433

__b)__ What is the MI(Label, F1 \| F3 = True) in the synthetic_v2_1000.dat?

Simulated MI(Label, F1 \| F3 = True) = 0.0582666


__Question 5:__ 

__a)__ What is the MI(Label, F1 \| F3 = False) according to the Bayesian network structure and parameters? Fill in the following tables provide your final answer.

P(X | Y) = P(X,Y)/P(Y)

P(Label, F1| F3) = P(Label,F1,F3)/P(F3) 

                 = P(F3)P(F1|F3)P(Label|F1,F3)/P(F3) 
                      
                 = P(F1|F3=FALSE)P(Label|F1,F3=FALSE)

| Label | F1 | P(Label, F1 \| F3 = False) |
| --- | --- | --- |
| A | True | (0.0604)(0.8182) = 0.04941928 |
| B | True | (0.0604)(0.1818) = 0.01098072 |
| A | False | (0.9396)(0.1111) = 0.10438956 |
| B | False | (0.9396)(0.8889) = 0.83521044 |

| Label | P(Label \| F3 = False) |
| --- | --- |
| A | 0.1538 |
| B | 0.8462 |

| F1 | P(F1 \| F3 = False) |
| --- | --- |
| True | 0.0604 |
| False | 0.9396 |

MI(Label, F1 \| F3 = False) 

                 = Sum( P(Label, F1| F3 = False) log(P(Label, F1| F3 = False)/(P(Label)P(F1| F3 = False))))

                 = P(A, True| False)log(P(A, True| False)/((P(A| False)P(True| False)))
              
                   + P(B, True| False)log(P(B, True| False)/((P(B| False)P(True| False)))
                
                   + P(A, False| False)log(P(A, False| False)/((P(A| False)P(False| False)))
             
                   + P(B, False| False)log(P(B, False| False)/((P(B| False)P(False| False)))
              
                 = 0.10514666759156843

__b)__ What is the MI(Label, F1 \| F3 = False) in the synthetic_v2_1000.dat? 

Simulated MI(Label, F1 \| F3 = False) = 0.11679318

__Question 6:__ 

__a)__ What is the MI(Label, F3 \| F1 = True) according to the Bayesian network structure and parameters? Fill in the following tables provide your final answer.

P(X | Y) = P(X,Y)/P(Y)

P(Label, F3| F1) = P(Label,F1,F3)/P(F1) 

                 = P(F1)P(F3|F1)P(Label|F1,F3)/P(F1) 
                      
                 = P(F3|F1=TRUE)P(Label|F3,F1=TRUE)

| Label | F3 | P(Label, F3 \| F1 = True) |
| --- | --- | --- |
| A | True | (0.95)(0.8182) = 0.77729 |
| B | True | (0.95)(0.1818) = 0.17271 |
| A | False | (0.05)(0.8182) = 0.04091 |
| B | False | (0.05)(0.1818) = 0.00909 |

| Label | P(Label \| F1 = True) |
| --- | --- |
| A | 0.8182 |
| B | 0.1818 |

| F3 | P(F3 \| F1 = True) |
| --- | --- |
| True | 0.95 |
| False | 0.05 |

MI(Label, F3 \| F1 = True) 

                 = Sum( P(Label, F3| F1 = True) log(P(Label, F3| F1 = True)/(P(Label)P(F3| F1 = True))))

                 = P(A, True| True)log(P(A, True| True)/((P(A| True)P(True| True)))
              
                   + P(B, True| True)log(P(B, True| True)/((P(B| True)P(True| True)))
                
                   + P(A, False| True)log(P(A, False| True)/((P(A| True)P(False| True)))
             
                   + P(B, False| True)log(P(B, False| True)/((P(B| True)P(False| True)))
              
                 = 0.0

__b)__ What is the MI(Label, F3 \| F1 = True) in the synthetic_v2_1000.dat?

Simulated MI(Label, F3 \| F1 = True) = 0.00103186


__Question 7:__ 

__a)__ What is the MI(Label, F3 \| F1 = False) according to the Bayesian network structure and parameters? Fill in the following tables provide your final answer.

P(X | Y) = P(X,Y)/P(Y)

P(Label, F3| F1) = P(Label,F1,F3)/P(F1) 

                 = P(F1)P(F3|F1)P(Label|F1,F3)/P(F1) 
                      
                 = P(F3|F1=FALSE)P(Label|F3,F1=FALSE)

| Label | F3 | P(Label, F3 \| F1 = False) |
| --- | --- | --- |
| A | True | (0.1111)(0.05) = 0.005555 |
| B | True | (0.8889)(0.05) = 0.044445 |
| A | False | (0.1111)(0.95) = 0.105545 |
| B | False | (0.8889)(0.95) = 0.844455 |

| Label | P(Label \| F1 = False) |
| --- | --- |
| A | 0.1111 |
| B | 0.8889 |

| F1 | P(F3 \| F1 = False) |
| --- | --- |
| True | 0.05 |
| False | 0.95 |

MI(Label, F3 \| F1 = False) 

                 = Sum( P(Label, F3| F1 = False) log(P(Label, F3| F1 = False)/(P(Label)P(F3| F1 = False))))

                 = P(A, True| False)log(P(A, True| False)/((P(A| False)P(True| False)))
              
                   + P(B, True| False)log(P(B, True| False)/((P(B| False)P(True| False)))
                
                   + P(A, False| False)log(P(A, False| False)/((P(A| False)P(False| False)))
             
                   + P(B, False| False)log(P(B, False| False)/((P(B| False)P(False| False)))
              
                 = 0.0

__b)__ What is the MI(Label, F3 \| F1 = False) in the synthetic_v2_1000.dat? 

Simulated MI(Label, F3 \| F1 = False) = 0.00123679

__Question 8:__ 

__a)__ Rank the features according to their importance, from highest to lowest, according to your answers to questions 1, 2, and 3. Use the true (not simulated) MIs for your ranking. Note that in this question, we are considering each feature in isolation and we are not taking correlations among features into account.

    Ranking as the following:
        1. F1
        2. F3
        3. F2

__b)__ Is F3 ranked higher or lower than F2? Why or why not? Do you agree with this ranking? Why or why not?

    F3 is ranked higer than F2 since it offers a higher mutual information with respect to 
    Label than F2. I do not agree with the ranking since F3 is dependent on F1 and is not
    directly linked to Label with respect to the tree created in Hugin. Further dependency checks
    need to be applied to confirm this assumption (we are given the tree but soley with numbers,
    F3 looks better than F2)

__Question 9:__

How does observing F3 affect F1's importance? Compare real MIs from Questions 1, 4, and 5. Explain why we see this effect.

    Observing F3 lets us know what contribution this feature does to the importance of F1 with 
    respect to Label. When we observe Questions 4 (0.067) and 5 (0.105) with respect to Question 
    1 (0.397), it is evident that F3 contributes to the F1's importance as it is above 0; 
    (.067 + .105) = .172 / .397 = 43% of the information is due to F3. Looking at the real MIs 
    for these questions, the results are replicated and the probabilities about the same 
    (.01 off); this is due a dependence exsisting between F1 and Label and the sample being "good".

__Question 10:__

How does observing F1 affect F3's importance? Compare real MIs from Questions 3, 6, and 7. Explain why we see this effect.

    Observing F1 lets us know what contribution this feature does to the importance of F3 
    with respect to Label. When we observe Question 6 and 7, it probabilities are 0 since 
    log(1) = 0 and all the 0s sum to 0. This means P(Label,F3|F1) = P(Label|F1)*P(F3|F1) and 
    concludes both Label and F3 are independent (conditional independence). Looking at the real 
    MIs for these questions, the results are almost replicated; even though the probabilities 
    are close between real and true, there exists an error since 0 should mean 0. The real 
    sample shows some dependence between F3 and Label due to uncertainty in the sampling but 
    the relation is minor to the point of accepting mutual independence.