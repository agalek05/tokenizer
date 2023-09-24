import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Wczytaj wytrenowany model t5 small i tokenizator
model_path = "./t5-small-spider"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Funkcja do tlumaczenia zapytan z jezyka naturalnego na jezyk SQL

def translate_to_sql(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors ="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length =512, num_return_sequences =1)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens =True)
    return decoded_output
# Przyklady z jezyka naturalnego

example_queries =[
    "Retrieve the names of all employees in the 'HR' department",
    "Show me the titles of all movies released before 2000.",
    "Find the total number of employees with a salary above 50000."
]
# Tlumaczenie i wyswietlenie zapytan SQL

for query in example_queries:
    sql_query = translate_to_sql(query, model, tokenizer)
    print("Zapytanie uzytkownika:", query)
    print("Wygenerowane zapytanie SQL:", sql_query)
    print()



