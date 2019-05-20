# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


OPERATION = ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate", "Solarize", "Posterize", "Contrast", "Color", "Brightness", "Sharpness", "AutoContrast", "Invert", "Equalize"]
exp0s = list()

def construct_good_policies(policies):
  """AutoAugment policies found on Cifar."""
  global exp0s
  length = len(policies)
  for i in range(length//5):
      a = []
      sub_policy = policies[i*5:(i+1)*5]
      operations = sub_policy[0]
      for j in range(2):
          if j == 0:
              operation_index = operations // 14
          else:
              operation_index = operations % 14
          operation_name = OPERATION[int(operation_index)]
          operation_pro = sub_policy[2*j+1]
          operation_mag = sub_policy[2*j+2]
          b = (operation_name, operation_pro, operation_mag)
          a.append(b)
      exp0s.append(a)

def good_policies():
    return exp0s

def delete_exp0s():
    global exp0s
    exp0s = []