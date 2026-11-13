import struct
import numpy as np
from transformers import GPT2LMHeadModel

# part 1: load the model
m = GPT2LMHeadModel.from_pretrained("gpt2")
params = list(m.named_parameters())  # all 148 (name, tensor) pairs

# part 2: onen ONE file, write how many tensors
f = open("model/gpt2.bin", "wb")  # "wb" = write binary
f.write(struct.pack("i", len(params)))  # 148, as a 4-byte int

# part 3: for each tensor, write name / shape / data
for n, p in params:
    name = n.encode()  # str->bytes
    f.write(struct.pack("i", len(name)))
    f.write(name)  # name itself

    f.write(struct.pack("i", p.dim()))  # munber of dims (1 or 2)
    for s in p.shape:
        f.write(struct.pack("i", s))  # each dim size

    p.detach().numpy().astype(np.float32).tofile(f)  # raw floats

f.close()
print("done")
