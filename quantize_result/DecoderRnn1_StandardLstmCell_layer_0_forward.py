# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class DecoderRnn1_StandardLstmCell_layer_0_forward(torch.nn.Module):
    def __init__(self):
        super(DecoderRnn1_StandardLstmCell_layer_0_forward, self).__init__()
        self.module_24 = py_nndct.nn.Input() #DecoderRnn1_StandardLstmCell_layer_0_forward::input_0
        self.module_25 = py_nndct.nn.Input() #DecoderRnn1_StandardLstmCell_layer_0_forward::input_1
        self.module_26 = py_nndct.nn.Input() #DecoderRnn1_StandardLstmCell_layer_0_forward::input_2
        self.module_27 = py_nndct.nn.Linear(in_features=16, out_features=32, bias=True) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_ii * input_0 + bias_i
        self.module_28 = py_nndct.nn.Linear(in_features=32, out_features=32, bias=False) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_hi * input_1
        self.module_29 = py_nndct.nn.Add() #DecoderRnn1_StandardLstmCell_layer_0_forward::y_i
        self.module_30 = py_nndct.nn.Linear(in_features=16, out_features=32, bias=True) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_if * input_0 + bias_f
        self.module_31 = py_nndct.nn.Linear(in_features=32, out_features=32, bias=False) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_hf * input_1
        self.module_32 = py_nndct.nn.Add() #DecoderRnn1_StandardLstmCell_layer_0_forward::y_f
        self.module_33 = py_nndct.nn.Linear(in_features=16, out_features=32, bias=True) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_ig * input_0 + bias_g
        self.module_34 = py_nndct.nn.Linear(in_features=32, out_features=32, bias=False) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_hg * input_1
        self.module_35 = py_nndct.nn.Add() #DecoderRnn1_StandardLstmCell_layer_0_forward::y_g
        self.module_36 = py_nndct.nn.Linear(in_features=16, out_features=32, bias=True) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_io * input_0 + bias_o
        self.module_37 = py_nndct.nn.Linear(in_features=32, out_features=32, bias=False) #DecoderRnn1_StandardLstmCell_layer_0_forward::w_ho * input_1
        self.module_38 = py_nndct.nn.Add() #DecoderRnn1_StandardLstmCell_layer_0_forward::y_o
        self.module_39 = py_nndct.nn.Sigmoid() #DecoderRnn1_StandardLstmCell_layer_0_forward::it
        self.module_40 = py_nndct.nn.Sigmoid() #DecoderRnn1_StandardLstmCell_layer_0_forward::ft
        self.module_41 = py_nndct.nn.Tanh() #DecoderRnn1_StandardLstmCell_layer_0_forward::cct
        self.module_42 = py_nndct.nn.Sigmoid() #DecoderRnn1_StandardLstmCell_layer_0_forward::ot
        self.module_43 = py_nndct.nn.Module('elemwise_mul') #DecoderRnn1_StandardLstmCell_layer_0_forward::it*cct
        self.module_44 = py_nndct.nn.Module('elemwise_mul') #DecoderRnn1_StandardLstmCell_layer_0_forward::ft*input_2
        self.module_45 = py_nndct.nn.Add() #DecoderRnn1_StandardLstmCell_layer_0_forward::c_next
        self.module_46 = py_nndct.nn.Tanh() #DecoderRnn1_StandardLstmCell_layer_0_forward::c_temp
        self.module_47 = py_nndct.nn.Module('elemwise_mul') #DecoderRnn1_StandardLstmCell_layer_0_forward::h_next

    def forward(self, *args):
        self.output_module_24 = self.module_24(input=args[0])
        self.output_module_25 = self.module_25(input=args[1])
        self.output_module_26 = self.module_26(input=args[2])
        self.output_module_27 = self.module_27(self.output_module_24)
        self.output_module_28 = self.module_28(self.output_module_25)
        self.output_module_29 = self.module_29(input=self.output_module_27, other=self.output_module_28)
        self.output_module_30 = self.module_30(self.output_module_24)
        self.output_module_31 = self.module_31(self.output_module_25)
        self.output_module_32 = self.module_32(input=self.output_module_30, other=self.output_module_31)
        self.output_module_33 = self.module_33(self.output_module_24)
        self.output_module_34 = self.module_34(self.output_module_25)
        self.output_module_35 = self.module_35(input=self.output_module_33, other=self.output_module_34)
        self.output_module_36 = self.module_36(self.output_module_24)
        self.output_module_37 = self.module_37(self.output_module_25)
        self.output_module_38 = self.module_38(input=self.output_module_36, other=self.output_module_37)
        self.output_module_39 = self.module_39(self.output_module_29)
        self.output_module_40 = self.module_40(self.output_module_32)
        self.output_module_41 = self.module_41(self.output_module_35)
        self.output_module_42 = self.module_42(self.output_module_38)
        self.output_module_43 = self.module_43(input=self.output_module_39, other=self.output_module_41)
        self.output_module_44 = self.module_44(input=self.output_module_40, other=self.output_module_26)
        self.output_module_45 = self.module_45(input=self.output_module_43, other=self.output_module_44)
        self.output_module_46 = self.module_46(self.output_module_45)
        self.output_module_47 = self.module_47(input=self.output_module_42, other=self.output_module_46)
        return self.output_module_47,self.output_module_45
