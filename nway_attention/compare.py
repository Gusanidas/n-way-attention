from examples.cfgs import Config
from trittention_cube import TrittentionCube
from trittention import Trittention
from attention import Attention
from transformer_models import TransformerBlock, Transformer, TriformerCube, Triformer, TriformerBlock, TriformerCubeBlock


import time
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_student_with_teacher(student_model, teacher_model, hidden_size, num_epochs=2500, batch_size=256, seq_len=42, learning_rate=0.002):
    t0 = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    # Initialize cosine scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    student_model.to(device)
    teacher_model.to(device)

    print(f"Count parameters student = {count_parameters(student_model)}")
    print(f"Count parameters teacher = {count_parameters(teacher_model)}")
    print(f"teacher = {teacher_model}")
    print(f"student_model = {student_model}")

    for epoch in range(num_epochs):
        inputs = torch.randn(batch_size, seq_len, hidden_size).to(device)
        #inputs = torch.randint(0, hidden_size, (batch_size, seq_len))

        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        student_outputs = student_model(inputs)

        loss = criterion(student_outputs, teacher_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.8f}, time = {time.time()-t0}')

model_cfg = Config(
    d_model=256,
    debug=True,
    layer_norm_eps=1e-5,
    d_vocab=8192,
    init_range=0.02,
    n_ctx=42,
    d_head=32,           
    d_mlp=288,          
    n_heads=8,         
    n_layers=3,       
    mlp_type="all",
    order_attn = True,
    attn_eq=True,
    with_ln=True,
)

#a1 = Attention(model_cfg)
#a2 = Attention(model_cfg)
#tc1 = TrittentionCube(model_cfg)
#tc2 = TrittentionCube(model_cfg)
#t1 = Trittention(model_cfg)
#t2 = Trittention(model_cfg)
#tb1 = TransformerBlock(model_cfg)
#stacked_attention = nn.Sequential(Attention(model_cfg), Attention(model_cfg))

stb1 = nn.Sequential(*[TransformerBlock(model_cfg, has_mlp=False) for _ in range(4)])
stri = nn.Sequential(*[TriformerBlock(model_cfg, has_mlp=False) for _ in range(2)])
train_student_with_teacher(stri, stb1, 256, num_epochs=702)
stb1 = nn.Sequential(*[TransformerBlock(model_cfg, has_mlp=False) for _ in range(4)])
stri = nn.Sequential(*[TriformerBlock(model_cfg, has_mlp=False) for _ in range(2)])
train_student_with_teacher(stb1, stri, 256)