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
2. `REDUCE-L/R-X`: pop the top two constituents of the stack, combine them into a new constituent with label `X`, and push the new constituent onto the stack.
3. `UNARY-X`: pop the top constituent off the stack, raise it to a new constituent with label `X`, and push the new constituent onto the stack.
4. `FINISH`: pop the root node of the stack and end parsing.

The deductive system of this method is as follows.

|Action|Before|After|
| ---:       | :---  |  :---  |
|`SHIFT`| `([s], i, false)` | `([s,w_i], i+1, false)`|
|`REDUCE-L/R-X`| `([s,s_2,s_1], i, false)` | `([s,X_{s_2,s_1}], i, false)` |
|`UNARY-X`| `([s,s_0], i, false)`| `([s, X_{s_0}], i, false)` |
|`FINISH`| `(s,i,false)` | `(s, i, true)`|

