# Transition-based Constituency Parsing (TCP)

## Introduction

This module implements bottom-up transition-based constituency parsing algorithms. 

Such an algorithm takes a left-to-right scan of the input sentence, maintaining two data structure: a stack and a buffer. 
- The stack stores partially constructed phrase structures. 
- The buffer stores input words. 

A state is defined as a triple `(s, i, f)` where `s` is the stack, `i` is the front index of the buffer (an integer), and `f` is a flag value (boolean) showing that the parsing is terminated. At each step, a transition action is applied to consume an input word or construct a new phrase structure. 

Different parsing methods employ their own sets of actions.

## Bottom-up Method

Given a state, the set of transition actions are:

1. `SHIFT`: pop the front word from the buffer, and push it onto the stack.
2. `REDUCE-L/R-X`: pop the top two constituents off the stack, combine them into a new constituent with label `X`, and push the new constituent onto the stack.
3. `UNARY-X`: pop the top constituent off the stack, raise it to a new constituent with label `X`, and push the new constituent onto the stack.
4. `FINISH`: pop the root node of the stack and end parsing.

The deductive system of this method is as follows.

|Action|Before|After|
| ---:       | :---  |  :---  |
|`SHIFT`| `([s], i, false)` | `([s,w_i], i+1, false)`|
|`REDUCE-L/R-X`| `([s,s_2,s_1], i, false)` | `([s,X(s_2,s_1)], i, false)` |
|`UNARY-X`| `([s,s_0], i, false)`| `([s, X(s_0)], i, false)` |
|`FINISH`| `(s,i,false)` | `(s, i, true)`|

For example, given the sentence `"the little boy likes red potatoes."` whose syntactic tree is 
```
(S 
    (NP (the little boy)) 
    (VP likes (NP (red potatoes)))
    (.)
)
```
Its corresponding binarized tree is
```
(S-r
    (NP-r 
        the 
        (NP-r* 
            little 
            boy
        )
    )
    (S-l* 
        (VP-l 
            likes 
            (NP-r* 
                red 
                tomatoes
            )
        (.)
    )
)
```
The sequence of actions to construct this binarized tree is as follows.

```
01. SHIFT
02. SHIFT
03. SHIFT
04. REDUCE-R-NP
05. REDUCE-R-NP
06. SHIFT
07. SHIFT
08. SHIFT
09. REDUCE-R-NP
10. REDUCE-L-VP
11. SHIFT
12. REDUCE-L-S
13. REDUCE-R-S
14. FINISH
```

| stack |  buffer | action | 
| ---:       | :---   | :---:    |
|  [] | [`the`, `little`, `boy`, `likes`,...] | `SHIFT`| 
|  [`the`] | [`little`, `boy`, `likes`,...] | `SHIFT`| 
|  [`the`, `little`] | [`boy`, `likes`, `red`,...] | `SHIFT`| 
|  [`the`, `little`, `boy`] | [`likes`, `red`, `potatoes`,...] | `REDUCE-R-NP`| 
|  [`the`, `NP-r`(`little`, `boy`)] | [`likes`, `red`, `potatoes`,...] | `REDUCE-R-NP`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`))] | [`likes`, `red`, `potatoes`,...] | `SHIFT`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `likes`] | [`red`, `potatoes`,`.`] | `SHIFT`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `likes`, `red`] | [`potatoes`,`.`] | `SHIFT`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `likes`, `red`, `potatoes`] | [`.`] | `REDUCE-R-NP`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `likes`, `NP-R`(`red`, `potatoes`)] | [`.`] | `REDUCE-L-VP`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `VP-L`(`likes`, `NP-R`(`red`, `potatoes`))] | [`.`] | `SHIFT`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `VP-L`(`likes`, `NP-R`(`red`, `potatoes`)), `.`] | [] | `REDUCE-L-S`| 
|  [`NP-R`(`the`, `NP-R`(`little`, `boy`)), `S-L`(`VP-L`(`likes`, `NP-R`(`red`, `potatoes`)), `.`)] | [] | `REDUCE-R-S`| 
|  [`S-R`(`NP-R`(`the`, `NP-R`(`little`, `boy`)), `S-L`(`VP-L`(`likes`, `NP-R`(`red`, `potatoes`)), `.`))] | [] | `FINISH`| 