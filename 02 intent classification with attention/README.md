# MerryUp_RE2RNN
combine RE2RNN with Merryup method and train together
## data processing
  1st get tokens and tags  
  2nd get all_tokens and all_tags  
  3rd get dict(w2i, i2w, t2i, i2t)  
  4th add padding  
  5th create dataset  
  6th create dataloader  
## model
use bilstm and glove emeds  
hidden2tag layer  
out with softmax  
remember to use forward and predict
