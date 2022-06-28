# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class EncoderRnn1_StandardLstmCell_layer_0_forward(torch.nn.Module):
    def __init__(self):
        super(EncoderRnn1_StandardLstmCell_layer_0_forward, self).__init__()
        self.module_0 = py_nndct.nn.Input() #EncoderRnn1_StandardLstmCell_layer_0_forward::input_0
        self.module_1 = py_nndct.nn.Input() #EncoderRnn1_StandardLstmCell_layer_0_forward::input_1
        self.module_2 = py_nndct.nn.Input() #EncoderRnn1_StandardLstmCell_layer_0_forward::input_2
        self.module_3 = py_nndct.nn.Linear(in_features=1, out_features=16, bias=True) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_ii * input_0 + bias_i
        self.module_4 = py_nndct.nn.Linear(in_features=16, out_features=16, bias=False) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_hi * input_1
        self.module_5 = py_nndct.nn.Add() #EncoderRnn1_StandardLstmCell_layer_0_forward::y_i
        self.module_6 = py_nndct.nn.Linear(in_features=1, out_features=16, bias=True) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_if * input_0 + bias_f
        self.module_7 = py_nndct.nn.Linear(in_features=16, out_features=16, bias=False) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_hf * input_1
        self.module_8 = py_nndct.nn.Add() #EncoderRnn1_StandardLstmCell_layer_0_forward::y_f
        self.module_9 = py_nndct.nn.Linear(in_features=1, out_features=16, bias=True) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_ig * input_0 + bias_g
        self.module_10 = py_nndct.nn.Linear(in_features=16, out_features=16, bias=False) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_hg * input_1
        self.module_11 = py_nndct.nn.Add() #EncoderRnn1_StandardLstmCell_layer_0_forward::y_g
        self.module_12 = py_nndct.nn.Linear(in_features=1, out_features=16, bias=True) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_io * input_0 + bias_o
        self.module_13 = py_nndct.nn.Linear(in_features=16, out_features=16, bias=False) #EncoderRnn1_StandardLstmCell_layer_0_forward::w_ho * input_1
        self.module_14 = py_nndct.nn.Add() #EncoderRnn1_StandardLstmCell_layer_0_forward::y_o
        self.module_15 = py_nndct.nn.Sigmoid() #EncoderRnn1_StandardLstmCell_layer_0_forward::it
        self.module_16 = py_nndct.nn.Sigmoid() #EncoderRnn1_StandardLstmCell_layer_0_forward::ft
        self.module_17 = py_nndct.nn.Tanh() #EncoderRnn1_StandardLstmCell_layer_0_forward::cct
        self.module_18 = py_nndct.nn.Sigmoid() #EncoderRnn1_StandardLstmCell_layer_0_forward::ot
        self.module_19 = py_nndct.nn.Module('elemwise_mul') #EncoderRnn1_StandardLstmCell_layer_0_forward::it*cct
        self.module_20 = py_nndct.nn.Module('elemwise_mul') #EncoderRnn1_StandardLstmCell_layer_0_forward::ft*input_2
        self.module_21 = py_nndct.nn.Add() #EncoderRnn1_StandardLstmCell_layer_0_forward::c_next
        self.module_22 = py_nndct.nn.Tanh() #EncoderRnn1_StandardLstmCell_layer_0_forward::c_temp
        self.module_23 = py_nndct.nn.Module('elemwise_mul') #EncoderRnn1_StandardLstmCell_layer_0_forward::h_next

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(input=args[1])
        self.output_module_2 = self.module_2(input=args[2])
        self.output_module_3 = self.module_3(self.output_module_0)
        self.output_module_4 = self.module_4(self.output_module_1)
        self.output_module_5 = self.module_5(input=self.output_module_3, other=self.output_module_4)
        self.output_module_6 = self.module_6(self.output_module_0)
        self.output_module_7 = self.module_7(self.output_module_1)
        self.output_module_8 = self.module_8(input=self.output_module_6, other=self.output_module_7)
        self.output_module_9 = self.module_9(self.output_module_0)
        self.output_module_10 = self.module_10(self.output_module_1)
        self.output_module_11 = self.module_11(input=self.output_module_9, other=self.output_module_10)
        self.output_module_12 = self.module_12(self.output_module_0)
        self.output_module_13 = self.module_13(self.output_module_1)
        self.output_module_14 = self.module_14(input=self.output_module_12, other=self.output_module_13)
        self.output_module_15 = self.module_15(self.output_module_5)
        self.output_module_16 = self.module_16(self.output_module_8)
        self.output_module_17 = self.module_17(self.output_module_11)
        self.output_module_18 = self.module_18(self.output_module_14)
        self.output_module_19 = self.module_19(input=self.output_module_15, other=self.output_module_17)
        self.output_module_20 = self.module_20(input=self.output_module_16, other=self.output_module_2)
        self.output_module_21 = self.module_21(input=self.output_module_19, other=self.output_module_20)
        self.output_module_22 = self.module_22(self.output_module_21)
        self.output_module_23 = self.module_23(input=self.output_module_18, other=self.output_module_22)
        return self.output_module_23,self.output_module_21
