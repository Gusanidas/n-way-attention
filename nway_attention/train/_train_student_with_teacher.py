import time
import torch
from torch import nn, optim


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_student_with_teacher(
        student_model,
        teacher_model,
        hidden_size,
        num_epochs=2500,
        batch_size=256,
        seq_len=42,
        learning_rate=0.002,
        device='cpu'):
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
