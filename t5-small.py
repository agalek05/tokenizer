import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Wczytaj zestaw danych spider
dataset = load_dataset('spider')

# TokenizatorT5
tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Przygotuj dane treingowe i walidacyjne

def preprocess_data(examples):
    return tokenizer(examples['question'], examples['query'], truncation =True, padding ='max_length', max_length =512)
train_data = dataset['train'].map(preprocess_data, batched =True)
valid_data = dataset['validation'].map(preprocess_data, batched =True)

# Konfiguracja modelu t5-small
model =AutoModelForSeq2SeqLM.from_pretrained('t5-small')

# Ustaw urzadzenie na GPU, jesli jest dostepne
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Konfiguracja parametrow treningu
optimizer = torch.optim.AdamW(model.parameters(), lr =5e-5, )
num_epochs =3
batch_size =8

# Petla trenigowa
for epoch in range(num_epochs):
    model.train()
    total_loss =0

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        labels = torch.tensor(batch['input_ids']).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids =input_ids, attention_mask =attention_mask, labels =labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss /len(train_data)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average LOSS: {average_loss}")
# Zapisz wytrenowany model

model.save_pretrained('./t5-small-spider')
tokenizer.save_pretrained('./t5-small-spider')

# Oceniaj model na zestawie testowym
model.eval()
total_test_loss =0

for i in range(0, len(valid_data),batch_size):
    batch = valid_data[i:i + batch_size]
    input_ids = torch.tensor(batch['input_ids']).to(device)
    attention_mask = torch.tensor(batch['attention_mask']).to(device)
    labels = torch.tensor(batch['input_ids']).to(device)

    with torch.no_grad():
        outputs = model(input_ids =input_ids, attention_mask =attention_mask, labels =labels)
        loss = outputs.loss
        total_test_loss += loss.item()

average_test_loss = total_test_loss / len(valid_data)
print(f"Average valid loss: {average_test_loss}")
