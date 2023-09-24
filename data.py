import pandas as pd
import datasets

# Pobranie datasetu spider
dataset = datasets.load_dataset('spider')

# Konwersja danych do ramki danych (DataFrame)

df = pd.DataFrame(dataset['train'])

# Wyswietlenie przykladowych pytan SQL
sample_queries = df['query'].sample(5).tolist()
sample_sql_queries = df['query'].sample(5).tolist()

for i, (query, sql_query) in enumerate(zip(sample_queries, sample_sql_queries)):
    print(f"Przyklad {i}:")
    print(f"Zapytanie w jezyku naturalnym: {query}")
    print(f"Zapytanie w jezyku SQL: {sql_query}")
    print()
